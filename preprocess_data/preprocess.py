import argparse
import json
import math
import os
import pickle
import random
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_relative(path: str, data_root: str) -> str:
    if not path:
        return path
    if path.startswith(("http://", "https://", "s3://", "gs://", "oss://")):
        return path
    try:
        return os.path.relpath(path, data_root)
    except ValueError:
        return path


def json_dump(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def frame_paths_from_dir(video_dir: str, frame_names: Optional[list[str]] = None) -> list[str]:
    if frame_names:
        paths = []
        for name in frame_names:
            stem = str(name)
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = os.path.join(video_dir, stem if stem.endswith(ext) else f"{stem}{ext}")
                if os.path.exists(candidate):
                    paths.append(candidate)
                    break
        return paths

    if not os.path.isdir(video_dir):
        return []
    files = sorted(
        file
        for file in os.listdir(video_dir)
        if os.path.splitext(file.lower())[1] in IMAGE_EXTENSIONS
    )
    return [os.path.join(video_dir, file) for file in files]


def get_image_size(path: str, fallback: tuple[int, int] = (640, 480)) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return fallback


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": rle["counts"]}


def decode_rle(rle: Optional[dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(rle, dict) or "counts" not in rle:
        return None
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    try:
        return (mask_utils.decode(rle) > 0).astype(np.uint8)
    except Exception:
        return None


def coco_annotation_to_mask(annotation: dict[str, Any], height: int, width: int) -> Optional[np.ndarray]:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            if not polygon:
                continue
            points = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
        return mask if np.any(mask) else None

    if isinstance(segmentation, dict):
        return decode_rle(segmentation)

    return None


def sample_frame_indices(
    total_frames: int,
    visible_indices: list[int],
    max_frames: int,
) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= max_frames:
        return list(range(total_frames))

    sampled = set(np.linspace(0, total_frames - 1, max(1, max_frames // 3), dtype=int).tolist())
    if visible_indices:
        visible_take = min(len(visible_indices), max(1, max_frames // 2))
        visible_pick = np.linspace(0, len(visible_indices) - 1, visible_take, dtype=int).tolist()
        sampled.update(visible_indices[idx] for idx in visible_pick)
        sampled.update(
            idx
            for idx in (
                max(0, visible_indices[0] - 1),
                visible_indices[0],
                visible_indices[-1],
                min(total_frames - 1, visible_indices[-1] + 1),
            )
        )

    sampled = sorted(sampled)
    if len(sampled) > max_frames:
        sampled = [sampled[idx] for idx in np.linspace(0, len(sampled) - 1, max_frames, dtype=int)]
    return sampled


def visible_indices_from_mask_dict(mask_dict: dict[str, Any], anno_ids: list[str], total_frames: int) -> list[int]:
    visible = []
    for frame_idx in range(total_frames):
        for anno_id in anno_ids:
            masks = mask_dict.get(str(anno_id))
            if isinstance(masks, list) and frame_idx < len(masks) and masks[frame_idx] is not None:
                visible.append(frame_idx)
                break
    return visible


def resize_mask_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def union_masks(frame_masks: list[Optional[dict[str, Any]]]) -> Optional[np.ndarray]:
    decoded = [decode_rle(rle) for rle in frame_masks]
    decoded = [mask for mask in decoded if mask is not None and np.any(mask)]
    if not decoded:
        return None

    height = max(mask.shape[0] for mask in decoded)
    width = max(mask.shape[1] for mask in decoded)
    union = np.zeros((height, width), dtype=np.uint8)
    for mask in decoded:
        union |= resize_mask_like(mask, (height, width))
    return union


def normalized_point(x: int, y: int, width: int, height: int) -> list[int]:
    return [
        int(round(x * 1000 / max(width - 1, 1))),
        int(round(y * 1000 / max(height - 1, 1))),
    ]


def point_to_grid(point: list[int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    height, width = grid_shape
    x = int(point[0] / 1000.0 * width)
    y = int(point[1] / 1000.0 * height)
    return max(0, min(x, width - 1)), max(0, min(y, height - 1))


def sample_positive_points(mask: np.ndarray) -> list[list[int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [[500, 500], [500, 500], [500, 500]]

    cx = int(round(float(xs.mean())))
    cy = int(round(float(ys.mean())))
    candidates = [(cx, cy)]

    quantiles = (0.25, 0.75)
    order = np.argsort(xs)
    for q in quantiles:
        idx = int(q * (len(order) - 1))
        candidates.append((int(xs[order[idx]]), int(ys[order[idx]])))

    height, width = mask.shape
    points = [normalized_point(x, y, width, height) for x, y in candidates[:3]]
    while len(points) < 3:
        points.append(points[-1])
    return points


def dino_similarity_score(
    candidate: list[int],
    positive_points: list[list[int]],
    dino_grid: Optional[torch.Tensor],
) -> float:
    if dino_grid is None or dino_grid.numel() == 0:
        return 0.0
    dino_grid = F.normalize(dino_grid.float(), p=2, dim=-1)
    height, width = dino_grid.shape[:2]
    cx, cy = point_to_grid(candidate, (height, width))
    candidate_feat = dino_grid[cy, cx]
    best = -1.0
    for point in positive_points:
        px, py = point_to_grid(point, (height, width))
        best = max(best, float(torch.dot(candidate_feat, dino_grid[py, px]).item()))
    return max(0.0, best)


def sample_negative_points(
    mask: np.ndarray,
    positive_points: list[list[int]],
    dino_grid: Optional[torch.Tensor],
    rng: random.Random,
) -> list[list[int]]:
    height, width = mask.shape
    kernel_size = max(7, int(min(height, width) * 0.04) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    ring = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~(mask.astype(bool))
    ys, xs = np.where(ring)
    if len(xs) == 0:
        ys, xs = np.where(mask == 0)
    if len(xs) == 0:
        return [[0, 0], [1000, 0], [0, 1000]]

    candidate_indices = list(range(len(xs)))
    rng.shuffle(candidate_indices)
    candidate_indices = candidate_indices[: min(2048, len(candidate_indices))]

    dist_to_mask = cv2.distanceTransform((1 - mask.astype(np.uint8)), cv2.DIST_L2, 3)
    diagonal = math.sqrt(height * height + width * width)
    scored = []
    for idx in candidate_indices:
        x, y = int(xs[idx]), int(ys[idx])
        point = normalized_point(x, y, width, height)
        near_score = max(0.0, 1.0 - float(dist_to_mask[y, x]) / max(diagonal * 0.25, 1.0))
        sim_score = dino_similarity_score(point, positive_points, dino_grid)
        scored.append((0.6 * sim_score + 0.4 * near_score, point))

    scored.sort(key=lambda item: item[0], reverse=True)
    points = []
    min_gap = 80
    for _, point in scored:
        if all(abs(point[0] - prev[0]) + abs(point[1] - prev[1]) >= min_gap for prev in points):
            points.append(point)
        if len(points) == 3:
            break
    while len(points) < 3:
        points.append(scored[len(points) % len(scored)][1])
    return points


class DINOFeatureExtractor:
    def __init__(self, model_name: str, device: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        if not enabled:
            return

        from transformers import AutoImageProcessor, AutoModel

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        except Exception as exc:
            print(f"Warning: failed to load DINO model {model_name}: {exc}. Features will be skipped.")
            self.enabled = False
            self.processor = None
            self.model = None

    def extract_grid_features(self, image_path: str, grid_size: int = 37) -> Optional[torch.Tensor]:
        if not self.enabled or self.processor is None or self.model is None:
            return None

        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (grid_size * 14, grid_size * 14))
        inputs = self.processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            patch_tokens = outputs.last_hidden_state[0, 1:, :]
            grid = patch_tokens.view(grid_size, grid_size, -1)
            return F.normalize(grid, p=2, dim=-1).half().cpu()


def feature_tensor_for_frames(
    frame_paths: list[str],
    extractor: DINOFeatureExtractor,
) -> Optional[torch.Tensor]:
    features = []
    for frame_path in frame_paths:
        feature = extractor.extract_grid_features(frame_path)
        if feature is None:
            return None
        features.append(feature)
    return torch.stack(features, dim=0) if features else None


def choose_representative_frame(frame_masks: dict[str, list[Optional[dict[str, Any]]]]) -> tuple[int, np.ndarray]:
    num_frames = max((len(masks) for masks in frame_masks.values()), default=0)
    best_idx = 0
    best_mask = None
    best_area = -1
    for idx in range(num_frames):
        mask = union_masks([masks[idx] if idx < len(masks) else None for masks in frame_masks.values()])
        area = int(mask.sum()) if mask is not None else 0
        if area > best_area:
            best_idx = idx
            best_mask = mask
            best_area = area
    if best_mask is None:
        raise ValueError("sample has no visible mask")
    return best_idx, best_mask


def build_answer(data_type: str, time_value: float, positive_points: list[list[int]], negative_points: list[list[int]]) -> str:
    payload: dict[str, Any] = {
        "positive_points": positive_points,
        "negative_points": negative_points,
    }
    if data_type == "video":
        payload = {"time": round(float(time_value), 3), **payload}
    return f"<answer>{json.dumps(payload, separators=(',', ':'))}</answer>"


def build_problem(expression: str, data_type: str) -> str:
    media_token = "<image>" if data_type == "image" else "<video>"
    return f'{media_token}\nLocate the object described as: "{expression}".'


def sample_record(
    *,
    problem_id: int,
    data_root: str,
    data_source: str,
    expression: str,
    data_type: str,
    frame_paths: list[str],
    frame_masks: dict[str, list[Optional[dict[str, Any]]]],
    video_meta: dict[str, Any],
    feature_tensor: Optional[torch.Tensor],
    feature_output_path: Optional[str],
    rng: random.Random,
) -> dict[str, Any]:
    selected_idx, selected_mask = choose_representative_frame(frame_masks)
    selected_dino = None
    if feature_tensor is not None and feature_tensor.dim() == 4 and selected_idx < feature_tensor.shape[0]:
        selected_dino = feature_tensor[selected_idx]

    positive_points = sample_positive_points(selected_mask)
    negative_points = sample_negative_points(selected_mask, positive_points, selected_dino, rng)
    sampled_times = video_meta.get("sampled_times") or [0.0]
    answer = build_answer(data_type, sampled_times[selected_idx] if selected_idx < len(sampled_times) else 0.0, positive_points, negative_points)

    rel_frames = [to_relative(path, data_root) for path in frame_paths]
    rel_video_meta = dict(video_meta)
    rel_video_meta["sampled_frame_paths"] = rel_frames

    feature_rel = ""
    if feature_tensor is not None and feature_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(feature_output_path)), exist_ok=True)
        torch.save(feature_tensor, feature_output_path)
        feature_rel = to_relative(feature_output_path, data_root)

    record = {
        "problem_id": problem_id,
        "problem": build_problem(expression, data_type),
        "problem_type": "segmentation",
        "data_type": data_type,
        "data_source": data_source,
        "answer": answer,
        "reward_extra": {
            "video_meta": rel_video_meta,
            "masks": frame_masks,
            "feature_dict": feature_rel,
        },
    }
    if data_type == "image":
        record["images"] = rel_frames[:1]
    else:
        record["videos"] = [rel_frames]
    return record


def mask_sequence_from_dict(mask_dict: dict[str, Any], anno_ids: list[str], frame_indices: list[int]) -> dict[str, list[Optional[dict[str, Any]]]]:
    result = {}
    for anno_id in anno_ids:
        masks = mask_dict.get(str(anno_id), [])
        selected = []
        for idx in frame_indices:
            rle = masks[idx] if isinstance(masks, list) and idx < len(masks) else None
            if isinstance(rle, dict) and isinstance(rle.get("counts"), bytes):
                rle = dict(rle)
                rle["counts"] = rle["counts"].decode("utf-8")
            selected.append(rle)
        if any(item is not None for item in selected):
            result[str(anno_id)] = selected
    return result


def davis_mask(data_root: str, split: str, video_name: str, obj_id: int, frame_idx: int) -> Optional[np.ndarray]:
    path = os.path.join(data_root, "video_datas", "davis17", split, "Annotations", video_name, f"{frame_idx:05d}.png")
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[:, :, 2]
    color = 128 + (obj_id - 1) * 64
    mask = (image == color).astype(np.uint8)
    return mask if np.any(mask) else None


def process_ref_seg(data_root: str, dataset_name: str, max_frames: int) -> list[dict[str, Any]]:
    ref_root = os.path.join(data_root, "ref_seg", dataset_name)
    refs_name = "refs(unc).p" if dataset_name in {"refcoco", "refcoco+"} else "refs(google).p"
    refs_path = os.path.join(ref_root, refs_name)
    instances_path = os.path.join(ref_root, "instances.json")
    if not os.path.exists(refs_path) or not os.path.exists(instances_path):
        return []

    with open(refs_path, "rb") as f:
        refs = pickle.load(f)
    instances = load_json(instances_path)
    image_info = {image["id"]: image for image in instances.get("images", [])}
    annotations = {annotation["id"]: annotation for annotation in instances.get("annotations", [])}
    image_dir = os.path.join(ref_root, "coco2014", "train2014")

    samples = []
    for ref in refs:
        if "train" not in str(ref.get("split", "")):
            continue
        info = image_info.get(ref.get("image_id"))
        if info is None:
            continue
        image_path = os.path.join(image_dir, info["file_name"])
        if not os.path.exists(image_path):
            continue
        width = int(info.get("width", get_image_size(image_path)[0]))
        height = int(info.get("height", get_image_size(image_path)[1]))
        anno_ids = [str(anno_id) for anno_id in ref.get("anno_ids", [ref.get("ann_id")]) if anno_id is not None]
        frame_masks = {}
        for anno_id in anno_ids:
            annotation = annotations.get(int(anno_id))
            mask = coco_annotation_to_mask(annotation, height, width) if annotation else None
            if mask is not None:
                frame_masks[anno_id] = [encode_mask(mask)]
        if not frame_masks:
            continue
        for sentence in ref.get("sentences", []):
            expression = sentence.get("sent", "")
            if expression:
                samples.append(
                    {
                        "data_source": dataset_name.upper(),
                        "data_type": "image",
                        "expression": expression,
                        "frame_paths": [image_path],
                        "frame_masks": frame_masks,
                        "video_meta": {
                            "video": f"{dataset_name}_{ref.get('ref_id')}",
                            "fps": 1.0,
                            "height": height,
                            "width": width,
                            "target_anno_ids": anno_ids,
                            "sampled_frame_ids": [0],
                            "sampled_times": [0.0],
                            "sampled_frame_paths": [image_path],
                        },
                    }
                )
    return samples


def process_revos_like(
    data_root: str,
    dataset_name: str,
    split: str,
    fps: float,
    max_frames: int,
) -> list[dict[str, Any]]:
    dataset_root = os.path.join(data_root, "video_datas", dataset_name)
    if dataset_name == "revos":
        meta_path = os.path.join(dataset_root, f"meta_expressions_{split}_.json")
        frame_root = dataset_root
        mask_path = os.path.join(dataset_root, "mask_dict.json")
    else:
        meta_path = os.path.join(dataset_root, split, "meta_expressions.json")
        frame_root = os.path.join(dataset_root, split, "JPEGImages")
        mask_path = os.path.join(dataset_root, split, "mask_dict.json")

    if not os.path.exists(meta_path) or not os.path.exists(mask_path):
        return []

    meta = load_json(meta_path)
    mask_dict = load_json(mask_path)
    samples = []
    for video_name, video_info in meta.get("videos", {}).items():
        frames = video_info.get("frames", [])
        if dataset_name == "revos":
            video_dir = os.path.join(frame_root, *video_name.split("/"))
            if not os.path.isdir(video_dir) and "/" in video_name:
                video_dir = os.path.join(frame_root, video_name.split("/")[0], video_name.split("/")[-1])
        else:
            video_dir = os.path.join(frame_root, video_name)
        frame_paths = frame_paths_from_dir(video_dir, frames)
        if not frame_paths:
            continue
        width, height = get_image_size(frame_paths[0], fallback=(1280, 720))
        for exp_id, exp_info in video_info.get("expressions", {}).items():
            anno_ids = exp_info.get("anno_id", [])
            if not isinstance(anno_ids, list):
                anno_ids = [anno_ids]
            anno_ids = [str(anno_id) for anno_id in anno_ids]
            visible = visible_indices_from_mask_dict(mask_dict, anno_ids, len(frame_paths))
            sampled_indices = sample_frame_indices(len(frame_paths), visible, max_frames)
            sampled_paths = [frame_paths[idx] for idx in sampled_indices]
            frame_masks = mask_sequence_from_dict(mask_dict, anno_ids, sampled_indices)
            if not frame_masks:
                continue
            samples.append(
                {
                    "data_source": dataset_name.upper(),
                    "data_type": "video",
                    "expression": exp_info.get("exp", ""),
                    "frame_paths": sampled_paths,
                    "frame_masks": frame_masks,
                    "video_meta": {
                        "video": video_name,
                        "fps": fps,
                        "height": height,
                        "width": width,
                        "target_anno_ids": anno_ids,
                        "sampled_frame_ids": [int(idx) for idx in sampled_indices],
                        "sampled_times": [float(idx) / fps for idx in sampled_indices],
                        "sampled_frame_paths": sampled_paths,
                    },
                }
            )
    return samples


def process_davis17(data_root: str, split: str, max_frames: int) -> list[dict[str, Any]]:
    dataset_root = os.path.join(data_root, "video_datas", "davis17")
    meta_path = os.path.join(dataset_root, "meta_expressions", split, "meta_expressions.json")
    frame_root = os.path.join(dataset_root, split, "JPEGImages")
    if not os.path.exists(meta_path):
        return []

    meta = load_json(meta_path)
    samples = []
    fps = 24.0
    for video_name, video_info in meta.get("videos", {}).items():
        frames = video_info.get("frames", [])
        frame_paths = frame_paths_from_dir(os.path.join(frame_root, video_name), frames)
        if not frame_paths:
            continue
        width, height = get_image_size(frame_paths[0], fallback=(854, 480))
        for exp_id, exp_info in video_info.get("expressions", {}).items():
            obj_id = int(exp_info.get("obj_id", 1))
            visible = []
            for idx in range(len(frame_paths)):
                mask = davis_mask(data_root, split, video_name, obj_id, idx)
                if mask is not None:
                    visible.append(idx)
            sampled_indices = sample_frame_indices(len(frame_paths), visible, max_frames)
            sampled_paths = [frame_paths[idx] for idx in sampled_indices]
            rles = []
            for idx in sampled_indices:
                mask = davis_mask(data_root, split, video_name, obj_id, idx)
                rles.append(encode_mask(mask) if mask is not None else None)
            frame_masks = {str(obj_id): rles} if any(item is not None for item in rles) else {}
            if not frame_masks:
                continue
            samples.append(
                {
                    "data_source": "DAVIS17",
                    "data_type": "video",
                    "expression": exp_info.get("exp", ""),
                    "frame_paths": sampled_paths,
                    "frame_masks": frame_masks,
                    "video_meta": {
                        "video": video_name,
                        "fps": fps,
                        "height": height,
                        "width": width,
                        "target_anno_ids": [str(obj_id)],
                        "sampled_frame_ids": [int(idx) for idx in sampled_indices],
                        "sampled_times": [float(idx) / fps for idx in sampled_indices],
                        "sampled_frame_paths": sampled_paths,
                    },
                }
            )
    return samples


def process_rvos(data_root: str, split: str, max_frames: int) -> list[dict[str, Any]]:
    dataset_root = os.path.join(data_root, "video_datas", "rvos")
    meta_path = os.path.join(dataset_root, "meta_expressions", split, "meta_expressions.json")
    mask_path = os.path.join(dataset_root, "mask_dict.pkl")
    frame_root = os.path.join(dataset_root, split, "JPEGImages")
    if not os.path.exists(meta_path) or not os.path.exists(mask_path):
        return []

    meta = load_json(meta_path)
    with open(mask_path, "rb") as f:
        mask_dict = pickle.load(f)

    samples = []
    fps = 5.0
    global_idx = 0
    for video_name, video_info in meta.get("videos", {}).items():
        frames = video_info.get("frames", [])
        frame_paths = frame_paths_from_dir(os.path.join(frame_root, video_name), frames)
        if not frame_paths:
            continue
        width, height = get_image_size(frame_paths[0], fallback=(640, 480))
        for exp_id, exp_info in video_info.get("expressions", {}).items():
            anno_id = str(global_idx)
            mask_list = mask_dict.get(anno_id)
            global_idx += 1
            if not isinstance(mask_list, list):
                continue
            visible = [idx for idx, rle in enumerate(mask_list[: len(frame_paths)]) if rle is not None]
            sampled_indices = sample_frame_indices(len(frame_paths), visible, max_frames)
            sampled_paths = [frame_paths[idx] for idx in sampled_indices]
            frame_masks = mask_sequence_from_dict(mask_dict, [anno_id], sampled_indices)
            if not frame_masks:
                continue
            samples.append(
                {
                    "data_source": "RVOS",
                    "data_type": "video",
                    "expression": exp_info.get("exp", ""),
                    "frame_paths": sampled_paths,
                    "frame_masks": frame_masks,
                    "video_meta": {
                        "video": video_name,
                        "fps": fps,
                        "height": height,
                        "width": width,
                        "target_anno_ids": [anno_id],
                        "sampled_frame_ids": [int(idx) for idx in sampled_indices],
                        "sampled_times": [float(idx) / fps for idx in sampled_indices],
                        "sampled_frame_paths": sampled_paths,
                    },
                }
            )
    return samples


def collect_raw_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    for dataset in args.datasets:
        if dataset == "ref_seg":
            for ref_name in ("refcoco", "refcoco+", "refcocog"):
                samples.extend(process_ref_seg(args.data_root, ref_name, args.max_frames))
        elif dataset == "revos":
            samples.extend(process_revos_like(args.data_root, "revos", args.split, 5.0, args.max_frames))
        elif dataset == "mevis":
            samples.extend(process_revos_like(args.data_root, "mevis", args.split, 5.0, args.max_frames))
        elif dataset == "davis17":
            samples.extend(process_davis17(args.data_root, args.split, args.max_frames))
        elif dataset == "rvos":
            samples.extend(process_rvos(args.data_root, args.split, args.max_frames))
        else:
            print(f"Warning: unsupported dataset {dataset}, skipped.")
    return samples


def process_all_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    raw_samples = collect_raw_samples(args)
    rng.shuffle(raw_samples)
    if args.max_samples is not None:
        raw_samples = raw_samples[: args.max_samples]

    output_features = args.output_features
    if output_features is None:
        output_features = os.path.join(os.path.dirname(os.path.abspath(args.output_json)), "features")
    os.makedirs(output_features, exist_ok=True)

    extractor = DINOFeatureExtractor(args.dino_model, args.device, enabled=not args.skip_features)
    records = []
    for idx, raw in enumerate(tqdm(raw_samples, desc="Preprocessing NegPoint samples"), start=1):
        feature_tensor = feature_tensor_for_frames(raw["frame_paths"], extractor) if not args.skip_features else None
        feature_path = os.path.join(output_features, f"{raw['data_source'].lower()}_{idx}_feature.pt") if feature_tensor is not None else None
        try:
            records.append(
                sample_record(
                    problem_id=idx,
                    data_root=args.data_root,
                    data_source=raw["data_source"],
                    expression=raw["expression"],
                    data_type=raw["data_type"],
                    frame_paths=raw["frame_paths"],
                    frame_masks=raw["frame_masks"],
                    video_meta=raw["video_meta"],
                    feature_tensor=feature_tensor,
                    feature_output_path=feature_path,
                    rng=rng,
                )
            )
        except Exception as exc:
            print(f"Warning: failed to build sample {idx}: {exc}")

    json_dump(records, args.output_json)
    print(f"Saved {len(records)} samples to {args.output_json}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess NegPoint RL/SFT data")
    parser.add_argument("--data_root", type=str, default=os.environ.get("DATA_ROOT", "."))
    parser.add_argument("--output_json", type=str, default="preprocess_data/expression.json")
    parser.add_argument("--output_features", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["ref_seg", "revos", "davis17", "mevis", "rvos"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dino_model", type=str, default=os.environ.get("DINO_MODEL", "facebook/dinov2-small"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.data_root = os.path.abspath(args.data_root)
    args.output_json = os.path.abspath(args.output_json)
    if args.output_features is not None:
        args.output_features = os.path.abspath(args.output_features)
    return args


if __name__ == "__main__":
    process_all_samples(parse_args())
