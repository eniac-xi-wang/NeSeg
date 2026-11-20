"""Terminal GRPO reward, computed from worker-owned SAM 2 masks and annotations."""
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from pycocotools import mask as coco

from neseg.protocol import pixels

REWARD_NAME = "neseg"
REWARD_TYPE = "trace"


def encode(mask):
    rle = coco.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


@lru_cache(maxsize=2)
def feature_grids(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def affinity(grid, negatives, refs):
    # Only gather the requested DINO patches, even for long video sequences.
    h, w = grid.shape[:2]
    def vectors(points):
        idx = np.minimum((np.asarray(points) * [w, h] / 1000).astype(int), [w - 1, h - 1])
        v = np.asarray(grid[idx[:, 1], idx[:, 0]], dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    return (vectors(negatives) @ vectors(refs).T).max(axis=1).clip(0, 1)


def point_scores(action, target, refs, grid):
    zero = dict(positive_hit=0., positive_distance=0., negative_hit=0., negative_dino=0.)
    if not refs or not target.any():
        return zero
    pos, neg = action['positive_points'], action['negative_points']
    px, py = pixels(pos, target.shape).T
    nx, ny = pixels(neg, target.shape).T
    outside = ~target[ny, nx]
    reference_xy = pixels(refs, target.shape)
    distances = np.linalg.norm(np.stack((px, py), axis=-1)[:, None] - reference_xy, axis=-1)
    scale = max(0.1 * math.hypot(*target.shape), 1.)
    dino = affinity(grid, neg, refs) if grid is not None else np.zeros(len(neg))
    return dict(
        positive_hit=float(target[py, px].mean()),
        positive_distance=float(np.exp(-distances.min(axis=1) / scale).mean()),
        negative_hit=float(np.where(outside, 1., -1.).mean()),
        negative_dino=float((dino * outside).mean()),
    )


def compute_score(trace, ground_truth, *, data_root=None, format_weight=0.05, iou_weight=0.50,
                  positive_hit_weight=0.10, positive_distance_weight=0.10,
                  negative_hit_weight=0.10, negative_dino_weight=0.10, efficiency_weight=0.05):
    truth = json.loads(ground_truth)
    components = dict(format=0., iou=0., positive_hit=0., positive_distance=0.,
                      negative_hit=0., negative_dino=0., efficiency=0.)
    calls = trace['calls']
    if not trace['valid'] or not calls:
        return dict(overall=0., **components)
    predictions = trace['masks']
    if len(predictions) != len(truth['masks']):
        raise ValueError('SAM 2 mask count differs from the annotated sequence')
    components['format'] = 1.
    grids = None
    if negative_dino_weight:
        if not truth['features']:
            raise ValueError('Missing DINOv2 features. Prepare features or set negative_dino_weight=0.')
        grids = feature_grids(str(Path(data_root or os.environ['DATA_ROOT'], truth['features'])))
    scores = []
    intersection = union = 0
    for idx, (ref, pred) in enumerate(zip(truth['masks'], predictions)):
        mask = coco.decode(pred).astype(bool)
        target = coco.decode(ref).astype(bool) if ref else np.zeros_like(mask)
        if target.shape != mask.shape:
            raise ValueError('SAM 2 and reference mask resolutions differ')
        intersection += np.count_nonzero(target & mask)
        union += np.count_nonzero(target | mask)
        for call in calls:
            if call['frame'] == idx:
                scores.append(point_scores(call, target, truth['references'][idx],
                                           grids[idx] if grids is not None else None))
    for name in scores[0]:
        components[name] = float(np.mean([s[name] for s in scores]))
    components['iou'] = intersection / max(union, 1)
    components['efficiency'] = components['iou'] / len(calls)
    total = (format_weight * components['format'] + iou_weight * components['iou']
             + positive_hit_weight * components['positive_hit']
             + positive_distance_weight * components['positive_distance']
             + negative_hit_weight * components['negative_hit']
             + negative_dino_weight * components['negative_dino']
             + efficiency_weight * components['efficiency'])
    return dict(overall=total, **components)
