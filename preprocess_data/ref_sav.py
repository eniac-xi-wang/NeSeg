"""Read Sa2VA referring expressions aligned to the official 6 fps SA-V masklets."""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
from pycocotools import mask as coco


def _asset(root, name):
    path = Path(name)
    if (root / path).is_file():
        return root / path
    # Some Ref-SAV releases repeat the shard directory before sav_train.
    parts = path.parts
    if 'sav_train' in parts:
        return root.joinpath(*parts[parts.index('sav_train'):])
    return root / path


def _extract(video, frame_ids, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f'{i:06}.jpg' for i in frame_ids]
    missing = [(i, p) for i, p in zip(frame_ids, paths) if not p.exists()]
    if missing:
        cap = cv2.VideoCapture(str(video))
        try:
            for idx, path in missing:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Missing frame {idx} in {video}')
                with NamedTemporaryFile(dir=output, suffix='.jpg', delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    if not cv2.imwrite(str(tmp_path), frame):
                        raise OSError(f'Cannot write {path}')
                    os.replace(tmp_path, path)
                finally:
                    tmp_path.unlink(missing_ok=True)
        finally:
            cap.release()
    return list(map(str, paths))


def ref_sav_samples(data_root, max_frames, sample_indices):
    base = Path(data_root, 'video_datas')
    expressions = base / 'Ref-SAV.json'
    if not expressions.exists():
        expressions = base / 'ref_sav' / 'Ref-SAV.json'
    if not expressions.exists():
        return []
    assets = base / 'sam_v_full'
    with expressions.open() as stream:
        videos = json.load(stream)
    samples = []
    for video_id, item in videos.items():
        video = _asset(assets, item['video_path'])
        with _asset(assets, item['anno_path']).open() as stream:
            annotations = json.load(stream)
        for obj, entry in item['objects'].items():
            caption = entry.get('formated') or entry.get('video_caption') or entry.get('image_caption')
            if not caption:
                continue
            sequence = [row[int(obj)] for row in annotations['masklet']]
            visible = [i for i, rle in enumerate(sequence) if rle and coco.area(rle) > 0]
            if not visible:
                continue
            indices = sample_indices(len(sequence), visible, max_frames)
            ids = [4 * i for i in indices]
            paths = _extract(video, ids, base / '.neseg_frames' / video.stem)
            masks = [sequence[i] for i in indices]
            height, width = sequence[visible[0]]['size']
            samples.append(dict(data_source='REF-SAV', data_type='video', expression=caption,
                                frame_paths=paths, frame_masks={obj: masks},
                                video_meta=dict(video=video.stem, fps=24., height=height, width=width,
                                                target_anno_ids=[obj], sampled_frame_ids=ids,
                                                sampled_times=[i / 24 for i in ids], sampled_frame_paths=paths)))
    return samples
