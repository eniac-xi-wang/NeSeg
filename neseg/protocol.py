"""The action and observation vocabulary shared by SFT, rollout and SAM 2."""
import json
import math

import numpy as np

INSTRUCTIONS = (
    'Return one JSON object per turn. To call SAM 2, use '
    '{"action":"segment","time":0.0,"positive_points":[[x,y]],"negative_points":[[x,y]]}. '
    'Choose a displayed time in seconds. Coordinates are integers from 0 to 1000, '
    'measured from the top left. Give 1 to 3 points of each kind. '
    'Positive points include the target. Negative points exclude distractors. '
    'After seeing the predicted mask, add points to correct it or return {"action":"finish"}.'
)


def make_prompt(description, times):
    frames = '\n'.join(f'{t:g}s <image>' for t in times)
    return f'{frames}\n{description}\n{INSTRUCTIONS}'


def pixels(points, shape):
    height, width = shape
    return np.rint(np.asarray(points) * [(width - 1) / 1000, (height - 1) / 1000]).astype(int)


def parse_action(text, times):
    action = json.loads(text)
    if action == {"action": "finish"}:
        return action
    fields = {"action", "time", "positive_points", "negative_points"}
    if not isinstance(action, dict) or set(action) != fields or action["action"] != "segment":
        raise ValueError("expected a segment or finish action")
    t = action["time"]
    if type(t) not in (int, float) or not math.isfinite(t):
        raise ValueError("time must be finite")
    idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
    if abs(times[idx] - t) > 0.001:
        raise ValueError("choose a displayed time")
    for name in ("positive_points", "negative_points"):
        pts = action[name]
        if not isinstance(pts, list) or not 1 <= len(pts) <= 3:
            raise ValueError("expected one to three points per label")
        for pt in pts:
            if (not isinstance(pt, list) or len(pt) != 2
                    or any(type(x) is not int or not 0 <= x <= 1000 for x in pt)):
                raise ValueError("invalid normalized coordinate")
    return dict(action, frame=idx)


def observation(mask, time):
    # Fixed-size occupancy map preserves holes and disconnected components.
    import cv2
    grid = cv2.resize(mask.astype(np.float32), (12, 12), interpolation=cv2.INTER_AREA)
    rows = [''.join('1' if v >= 0.5 else '0' for v in row) for row in grid]
    ys, xs = np.nonzero(mask)
    box = None
    if xs.size:
        scale = np.array([max(mask.shape[1] - 1, 1), max(mask.shape[0] - 1, 1)])
        box = np.rint(np.array([[xs.min(), ys.min()], [xs.max(), ys.max()]]) * 1000 / scale).astype(int).tolist()
    return 'SAM 2 mask (top to bottom, 1 = foreground): ' + json.dumps(
        {"time": time, "box": box, "rows": rows}, separators=(',', ':')
    )


def feedback_suffix(text):
    # Continuation after Qwen's assistant end-of-message token.
    return f'\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n'
