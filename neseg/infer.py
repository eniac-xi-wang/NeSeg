"""Run a trained policy and save one binary foreground PNG per sampled frame."""
import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from pycocotools import mask as coco
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from neseg.protocol import make_prompt
from neseg.rollout import play
from neseg.sft import encode_context


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--frames', nargs='+', required=True)
    parser.add_argument('--times', type=float, nargs='+', required=True)
    parser.add_argument('--expression', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-calls', type=int, default=3)
    args = parser.parse_args()
    times = [round(t, 3) for t in args.times]
    if (len(args.frames) != len(times) or times != sorted(set(times)) or args.max_calls < 1
            or any(not math.isfinite(t) or t < 0 for t in times)):
        parser.error('Supply one increasing timestamp per frame and a positive call limit.')
    paths = [str(Path(path).resolve(strict=True)) for path in args.frames]
    proc = AutoProcessor.from_pretrained(args.policy)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.policy, torch_dtype=dtype).to(device).eval()
    row = dict(problem=make_prompt(args.expression, times), history=[], images=paths)
    context = encode_context(proc, Path('.'), row).to(device)
    def generate(history, remaining):
        suffix = context.input_ids.new_tensor([history])
        ids = torch.cat([context.input_ids, suffix], dim=1)
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=context.pixel_values,
                             image_grid_thw=context.image_grid_thw, do_sample=False, max_new_tokens=min(192, remaining))
        return out[0, ids.shape[1]:].tolist()
    _, _, trace = play(generate([], 1536), media=dict(images=paths, times=times), tokenizer=proc.tokenizer,
                       generate=generate, budget=1536, max_calls=args.max_calls)
    if not trace['valid']:
        raise RuntimeError('The policy did not complete a valid segmentation episode.')
    args.output.mkdir(parents=True, exist_ok=True)
    for i, rle in enumerate(trace['masks']):
        Image.fromarray(coco.decode(rle) * 255).save(args.output / f'{i:06}.png')
    print(f"Saved {len(paths)} masks after {len(trace['calls'])} SAM 2 calls.")


if __name__ == '__main__':
    main()
