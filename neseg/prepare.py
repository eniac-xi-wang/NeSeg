"""Export clean policy inputs and private reward labels from NeSeg annotations."""
import argparse
import hashlib
import json
import os
import random
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from neseg.protocol import make_prompt, observation, parse_action, pixels
from neseg.reward import encode
from neseg.segmenter import session
from preprocess_data.preprocess import sample_random_positive_points, union_masks


def make_rows(record, root, rng, with_feedback=False):
    extra = record['reward_extra']
    meta = extra['video_meta']
    paths = meta['sampled_frame_paths']
    times = [round(t, 3) for t in meta['sampled_times']]
    if not paths or len(paths) != len(times) or len(set(times)) != len(times):
        raise ValueError('Expected one distinct timestamp per sampled frame')
    description = record['problem'].removeprefix('<video>').removeprefix('<image>').strip()
    problem = make_prompt(description, times)
    masks, refs = [], []
    objects = list(extra['masks'].values())
    if not objects or any(len(seq) != len(paths) for seq in objects):
        raise ValueError('Frame and mask sequence lengths differ')
    for i in range(len(paths)):
        target = union_masks([seq[i] for seq in objects])
        masks.append(target)
        refs.append(sample_random_positive_points(target, rng) if target is not None else [])
    raw_answer = record['answer'].removeprefix('<answer>').removesuffix('</answer>')
    calls = json.loads(raw_answer)['tool_calls']
    if not calls:
        raise ValueError('No supervised point prompts')
    examples, history = [], []
    with ExitStack() as stack:
        video = stack.enter_context(session([root / p for p in paths])) if with_feedback else None
        for call in calls:
            action = dict(call, action='segment', time=call.get('time', times[0]))
            parsed = parse_action(json.dumps(action), times)
            i = parsed['frame']
            target = masks[i]
            if target is None:
                raise ValueError('Supervised timestamp has no target mask')
            # Older files used a centroid that could fall outside a concave object.
            x, y = pixels(action['positive_points'], target.shape).T
            if not target[y, x].all():
                action['positive_points'] = refs[i]
                parsed['positive_points'] = refs[i]
            x, y = pixels(action['negative_points'], target.shape).T
            if target[y, x].any():
                from preprocess_data.preprocess import sample_negative_points
                from neseg.reward import feature_grids
                grid = feature_grids(str(root / extra['feature_dict']))[i] if extra.get('feature_dict') else None
                action['negative_points'] = sample_negative_points(target, action['positive_points'], grid, rng)
                parsed['negative_points'] = action['negative_points']
            refs[i] = [list(p) for p in dict.fromkeys(map(tuple, action['positive_points'] + refs[i]))]
            text = json.dumps(action, separators=(',', ':'))
            examples.append(dict(problem=problem, images=paths, history=list(history), target=text))
            if video:
                predicted = video.apply(parsed)
                history.extend([dict(role='assistant', content=text),
                                dict(role='user', content=observation(predicted[i], times[i]))])
        if video:
            examples.append(dict(problem=problem, images=paths, history=history, target='{"action":"finish"}'))
    truth = dict(masks=[encode(m) if m is not None else None for m in masks],
                 references=refs, features=extra.get('feature_dict', ''))
    rl = dict(problem=problem, images=paths, neseg_times=times,
              answer=json.dumps(truth, separators=(',', ':')))
    return examples, rl


def export(records, output, root, val_fraction, seed, with_feedback):
    if not 0 <= val_fraction < 1:
        raise ValueError('val_fraction must be in [0, 1)')
    output.mkdir(parents=True, exist_ok=True)
    counts = {'train': 0, 'val': 0, 'sft': 0}
    rng = random.Random(seed)
    with TemporaryDirectory(dir=output) as temp:
        with ExitStack() as stack:
            files = {k: stack.enter_context(open(Path(temp, k + '.jsonl'), 'w')) for k in counts}
            for record in records:
                meta = record['reward_extra']['video_meta']
                group = (meta['sampled_frame_paths'][0] if record['data_type'] == 'image'
                         else record['data_source'] + '/' + meta['video'])
                digest = hashlib.sha256(f'{seed}/{group}'.encode()).digest()
                split = 'val' if int.from_bytes(digest[:8], 'big') / 2**64 < val_fraction else 'train'
                examples, rl = make_rows(record, root, rng, with_feedback and split == 'train')
                files[split].write(json.dumps(rl) + '\n')
                counts[split] += 1
                if split == 'train':
                    for example in examples:
                        files['sft'].write(json.dumps(example) + '\n')
                        counts['sft'] += 1
        for key in counts:
            os.replace(Path(temp, key + '.jsonl'), output / (key + '.jsonl'))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.05)
    parser.add_argument('--with-feedback', action='store_true')
    args = parser.parse_args()
    with args.records.open() as stream:
        print(export(json.load(stream), args.output, args.data_root, args.val_fraction, args.seed, args.with_feedback))


if __name__ == '__main__':
    main()
