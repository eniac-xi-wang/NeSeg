"""A frozen SAM 2 video session on the exact frame sequence seen by Qwen."""
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from neseg.protocol import pixels


@lru_cache(maxsize=1)
def predictor():
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(
        os.environ.get('SAM2_CONFIG', 'configs/sam2.1/sam2.1_hiera_t.yaml'),
        os.environ['SAM2_CHECKPOINT'], device=os.environ.get('SAM2_DEVICE', 'cpu'),
        hydra_overrides_extra=["++model.add_all_frames_to_correct_as_cond=true"],
        apply_postprocessing=False,
    )


@contextmanager
def session(paths):
    with TemporaryDirectory(prefix='neseg-frames-') as directory, torch.inference_mode():
        for idx, path in enumerate(paths):
            dst = Path(directory, f'{idx:06}.jpg')
            if Path(path).suffix.lower() in {'.jpg', '.jpeg'}:
                dst.symlink_to(Path(path).resolve(strict=True))
            else:
                with Image.open(path) as image:
                    image.convert('RGB').save(dst)
        model = predictor()
        state = model.init_state(video_path=directory)
        yield VideoSession(model, state)


class VideoSession:
    def __init__(self, model, state):
        self.model = model
        self.state = state

    def apply(self, action):
        frame = action['frame']
        pos, neg = action['positive_points'], action['negative_points']
        xy = pixels(pos + neg, (self.state['video_height'], self.state['video_width']))
        self.model.add_new_points_or_box(
            self.state, frame_idx=frame, obj_id=1, points=xy,
            labels=[1] * len(pos) + [0] * len(neg), clear_old_points=False,
        )
        masks = {}
        for reverse in (False, True):
            for idx, _, logits in self.model.propagate_in_video(
                self.state, start_frame_idx=frame, reverse=reverse
            ):
                masks[idx] = (logits[0, 0] > 0).cpu().numpy()
        return [masks[i] for i in range(self.state['num_frames'])]
