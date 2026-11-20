"""One policy action at a time, with private tool state for each GRPO completion."""
from contextlib import ExitStack
from copy import copy

from neseg.protocol import feedback_suffix, observation, parse_action
from neseg.reward import encode
from neseg.segmenter import session


def play(first_tokens, *, media, tokenizer, generate, budget, max_calls, open_session=session):
    times = media['times']
    tokens, policy_mask, calls = [], [], []
    masks, tool = [], None
    end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    valid = False
    with ExitStack() as stack:
        action_tokens = first_tokens
        while True:
            if not action_tokens or len(tokens) + len(action_tokens) > budget:
                raise RuntimeError('Generation did not respect the token budget')
            tokens.extend(action_tokens)
            policy_mask.extend([1] * len(action_tokens))
            # Strip only the normal chat terminator. Special tokens within JSON are invalid.
            content = action_tokens[:-1] if action_tokens[-1] == end_id else action_tokens
            try:
                action = parse_action(tokenizer.decode(content, skip_special_tokens=False), times)
            except (ValueError, TypeError, KeyError, OverflowError):
                break
            if action['action'] == 'finish':
                valid = bool(calls)
                break
            if tool is None:
                tool = stack.enter_context(open_session(media['images']))
            masks = tool.apply(action)
            calls.append(action)
            if len(calls) == max_calls:
                valid = True
                break
            frame = action['frame']
            message = feedback_suffix(observation(masks[frame], times[frame]))
            feedback = tokenizer.encode(message, add_special_tokens=False)
            if action_tokens[-1] != end_id:
                feedback = [end_id, *feedback]
            if len(tokens) + len(feedback) >= budget:
                break
            tokens.extend(feedback)
            policy_mask.extend([0] * len(feedback))
            action_tokens = generate(tokens, budget - len(tokens))
    trace = dict(valid=valid, calls=calls, masks=[encode(m) for m in masks] if valid else [])
    return tokens, policy_mask, trace


def generate_episodes(worker, inputs, media, adapters):
    params = copy(worker.sampling_params)
    budget = worker.config.response_length
    params.max_tokens = min(192, budget)
    first = worker.inference_engine.generate(prompts=inputs, sampling_params=params,
                                             lora_request=adapters, use_tqdm=worker.use_tqdm)
    episodes = []
    for idx, request in enumerate(first):
        for sample_idx, completion in enumerate(request.outputs):
            def follow(history, remaining):
                next_params = copy(params)
                next_params.n = 1
                next_params.max_tokens = min(192, remaining)
                if params.seed is not None:
                    next_params.seed = params.seed + sample_idx + len(history)
                prompt = dict(inputs[idx], prompt_token_ids=inputs[idx]['prompt_token_ids'] + history)
                output = worker.inference_engine.generate(
                    prompts=[prompt], sampling_params=next_params,
                    lora_request=adapters[idx:idx + 1] if adapters else None, use_tqdm=False,
                )
                return list(output[0].outputs[0].token_ids)
            episodes.append(play(list(completion.token_ids), media=media[idx], tokenizer=worker.tokenizer,
                                 generate=follow, budget=budget, max_calls=worker.config.neseg_max_calls))
    return tuple(map(list, zip(*episodes)))
