import os
import json
import pickle
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
from PIL import Image
from tqdm import tqdm


# ==========================================
# 数据集路径配置
# ==========================================
REF_SEG_BASE = "/home/ma-user/sfs_turbo/qinianwang/datasets/Sa2VA-Training/ref_seg"
VIDEO_DATAS_BASE = "/home/ma-user/sfs_turbo/qinianwang/datasets/Sa2VA-Training/video_datas"

# ref_seg 数据集
REFCOCO_PATH = os.path.join(REF_SEG_BASE, "refcoco")
REFCOCO_PLUS_PATH = os.path.join(REF_SEG_BASE, "refcoco+")
REFCOCOG_PATH = os.path.join(REF_SEG_BASE, "refcocog")

# video_datas 数据集
REVOS_PATH = os.path.join(VIDEO_DATAS_BASE, "revos")
DAVIS17_PATH = os.path.join(VIDEO_DATAS_BASE, "davis17")
MEVIS_PATH = os.path.join(VIDEO_DATAS_BASE, "mevis")
RVOS_PATH = os.path.join(VIDEO_DATAS_BASE, "rvos")
REF_SAV_PATH = os.path.join(VIDEO_DATAS_BASE, "ref_sav")


# ==========================================
# Prompt 模板
# ==========================================
# 修改 negpoint/dataloader.py 中的 PROMPT_TEMPLATE
PROMPT_TEMPLATE = """You are a helpful assistant that helps to locate objects in videos based on textual descriptions.
Please answer the following question: {question}
Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.
Please choose ONE time within the object(in seconds). 
For EACH independent object, provide a pair consisting of one positive point (inside the object) and one negative point (outside but near the object).

Answer in the format: 
"<think>...</think><answer>{{\"time\": <time_in_seconds>, \"point_pairs\": [ {{\"positive\": [x1,y1], \"negative\": [nx1,ny1]}}, {{\"positive\": [x2,y2], \"negative\": [nx2,ny2]}} ]}}</answer>"""


# ==========================================
# 工具函数
# ==========================================
def sample_frames_uniform(frame_paths: List[str], max_frames: int = 64) -> tuple:
    """
    均匀采样帧，返回采样后的帧路径和原始帧索引
    """
    total_frames = len(frame_paths)
    if total_frames <= max_frames:
        return frame_paths, list(range(total_frames))
    
    # 均匀采样
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    sampled_paths = [frame_paths[i] for i in indices]
    return sampled_paths, indices.tolist()


def get_hybrid_sampled_frames(all_frames, target_anno_ids, mask_dict, max_frames=64):
    """
    智能混合采样策略：全局稀疏帧 + 目标可见区域高密帧 + 边界帧
    返回: 采样后的帧路径列表和原始帧索引 (去重且有序)
    """
    # 1. 找出包含目标的所有“有效帧”索引
    valid_frame_indices = []
    for i, frame_path in enumerate(all_frames):
        is_visible = False
        for aid in target_anno_ids:
            aid_str = str(aid)
            if aid_str in mask_dict and i < len(mask_dict[aid_str]):
                if mask_dict[aid_str][i] is not None:
                    is_visible = True
                    break
        if is_visible:
            valid_frame_indices.append(i)

    sampled_indices = set()

    # 策略 A：全局均匀采样 (约占 30% 预算，比如 20 帧) -> 保证全局时序认知
    num_global = min(len(all_frames), max_frames // 3)
    if num_global > 0:
        global_idx = np.linspace(0, len(all_frames) - 1, num_global, dtype=int)
        sampled_indices.update(global_idx)

    # 策略 B：可见区间均匀采样 (约占 50% 预算，比如 32 帧) -> 保证模型有足够的正样本可学
    if valid_frame_indices:
        num_visible = min(len(valid_frame_indices), max_frames // 2)
        visible_idx = np.linspace(0, len(valid_frame_indices) - 1, num_visible, dtype=int)
        sampled_indices.update([valid_frame_indices[i] for i in visible_idx])
        
        # 策略 C：边界帧 (首尾帧附近) -> 帮助模型学习“时间边界”
        first_visible = valid_frame_indices[0]
        last_visible = valid_frame_indices[-1]
        sampled_indices.update([max(0, first_visible - 1), first_visible, min(len(all_frames)-1, last_visible + 1), last_visible])

    # 如果还是超出了 max_frames，再做一次均匀降采样
    sampled_indices = sorted(list(sampled_indices))
    if len(sampled_indices) > max_frames:
        final_idx = np.linspace(0, len(sampled_indices) - 1, max_frames, dtype=int)
        sampled_indices = [sampled_indices[i] for i in final_idx]

    return [all_frames[i] for i in sampled_indices], sampled_indices


def get_frame_paths_from_video_dir(video_dir: str) -> List[str]:
    """
    从视频目录中获取所有帧的路径（按文件名排序）
    """
    if not os.path.exists(video_dir):
        return []
    
    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    return [os.path.join(video_dir, f) for f in frame_files]


def compute_sampled_times(frame_indices: List[int], fps: float = 5.0) -> List[float]:
    """
    根据帧索引和帧率计算对应的时间（秒）
    """
    return [idx / fps for idx in frame_indices]


# ==========================================
# ref_seg 数据集加载 (RefCOCO, RefCOCO+, RefCOCOg)
# ==========================================
def load_ref_dataset(ref_path: str, split: str = "train", dataset_name: str = "refcoco") -> List[Dict]:
    """
    加载 RefCOCO 系列数据集
    注意：ref_seg 是图像数据集，需要转换为视频格式（单帧）
    """
    samples = []
    
    # 加载 refs 文件 (pickle 格式)
    refs_file = os.path.join(ref_path, "refs(unc).p" if dataset_name in ["refcoco", "refcoco+"] else "refs(google).p")
    if not os.path.exists(refs_file):
        print(f"Warning: {refs_file} not found, skipping {dataset_name}")
        return samples
    
    with open(refs_file, 'rb') as f:
        refs_data = pickle.load(f)
    
    # 加载 instances.json 获取图像信息
    instances_file = os.path.join(ref_path, "instances.json")
    if not os.path.exists(instances_file):
        print(f"Warning: {instances_file} not found, skipping {dataset_name}")
        return samples
    
    with open(instances_file, 'r') as f:
        instances = json.load(f)
    
    # 构建图像 id 到文件名的映射
    image_id_to_file = {img['id']: img for img in instances['images']}
    
    # 构建 annotation id 到 mask 的映射
    anno_id_to_mask = {}
    for anno in instances['annotations']:
        anno_id_to_mask[anno['id']] = anno
    
    # COCO 图像根目录
    coco_image_dir = os.path.join(ref_path, "coco2014", "train2014")
    
    for ref in tqdm(refs_data, desc=f"Loading {dataset_name}"):
        # 只处理指定 split
        if split not in ref.get('split', ''):
            continue
        
        image_id = ref['image_id']
        anno_ids = ref.get('anno_ids', [ref.get('ann_id')])
        sentences = ref.get('sentences', [])
        
        if not sentences or not anno_ids:
            continue
        
        # 获取图像信息
        image_info = image_id_to_file.get(image_id)
        if not image_info:
            continue
        
        image_file = image_info['file_name']
        image_path = os.path.join(coco_image_dir, image_file)
        
        height = image_info.get('height', 480)
        width = image_info.get('width', 640)
        
        # 为每个 sentence 创建一个样本
        for sent in sentences:
            expression = sent.get('sent', '')
            
            # 构建 video_meta
            video_meta = {
                "video": f"{dataset_name}_{ref['ref_id']}",
                "fps": 1.0,  # 图像视为 1fps 的单帧视频
                "height": height,
                "width": width,
                "target_anno_ids": [str(aid) for aid in anno_ids],
                "sampled_frame_ids": [0],
                "sampled_times": [0.0],
                "sampled_frame_paths": [image_path]
            }
            
            # 构建 prompt
            prompt_text = PROMPT_TEMPLATE.format(question=f"Locate the {expression}.")
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": [image_path],
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            sample = {
                "prompt": prompt,
                "video_meta": video_meta
            }
            samples.append(sample)
    
    return samples


# ==========================================
# ReVOS 数据集加载
# ==========================================
def load_revos_dataset(split: str = "train") -> List[Dict]:
    """
    加载 ReVOS 数据集
    """
    samples = []
    
    # 加载 meta_expressions
    meta_file = os.path.join(REVOS_PATH, f"meta_expressions_{split}_.json")
    if not os.path.exists(meta_file):
        print(f"Warning: {meta_file} not found, skipping ReVOS")
        return samples
    
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    
    # 加载 mask_dict
    mask_dict_file = os.path.join(REVOS_PATH, "mask_dict.json")
    if not os.path.exists(mask_dict_file):
        print(f"Warning: {mask_dict_file} not found, skipping ReVOS")
        return samples
    
    with open(mask_dict_file, 'r') as f:
        mask_dict = json.load(f)
    
    # 视频根目录 - ReVOS 使用特殊的目录结构
    # 视频名格式如 "UVO/all/-CiPki3XuVI" 或 "LV-VIS/train/00005"
    
    videos = meta_data.get('videos', {})
    
    for video_name, video_info in videos.items():
        expressions = video_info.get('expressions', {})
        
        # video_name 格式可能是 "dataset/split/video_id" 或 "dataset/video_id"
        video_path_parts = video_name.split('/')
        
        # 构建视频帧目录路径
        video_dir = os.path.join(REVOS_PATH, *video_path_parts)
        
        if not os.path.exists(video_dir):
            # 尝试其他可能的目录结构
            if len(video_path_parts) >= 2:
                # 尝试 dataset/video_id (没有split层级)
                alt_video_dir = os.path.join(REVOS_PATH, video_path_parts[0], video_path_parts[-1])
                if os.path.exists(alt_video_dir):
                    video_dir = alt_video_dir
                else:
                    continue
            else:
                continue
        
        # 获取所有帧路径
        frame_paths = get_frame_paths_from_video_dir(video_dir)
        if not frame_paths:
            continue
        
        # 获取视频尺寸 (从第一帧)
        width, height = 720, 1280
        try:
            first_frame = next((p for p in frame_paths if os.path.exists(p)), None)
            if first_frame:
                with Image.open(first_frame) as img:
                    width, height = img.size
        except:
            pass
        
        for exp_id, exp_info in expressions.items():
            expression = exp_info.get('exp', '')
            anno_ids = exp_info.get('anno_id', [])
            
            if not isinstance(anno_ids, list):
                anno_ids = [anno_ids]
            
            # 转换为字符串格式的 anno_id
            target_anno_ids = [str(aid) for aid in anno_ids]
            
            # 使用混合采样策略
            sampled_paths, sampled_indices = get_hybrid_sampled_frames(
                frame_paths, target_anno_ids, mask_dict, max_frames=64
            )
            
            # 计算采样时间 (假设默认 fps=5)
            sampled_times = compute_sampled_times(sampled_indices, fps=5.0)
            
            # 构建 video_meta
            video_meta = {
                "video": video_name,
                "fps": 5.0,
                "height": height,
                "width": width,
                "target_anno_ids": target_anno_ids,
                "sampled_frame_ids": sampled_indices,
                "sampled_times": sampled_times,
                "sampled_frame_paths": sampled_paths
            }
            
            # 构建 prompt
            prompt_text = PROMPT_TEMPLATE.format(question=f"Locate the {expression}.")
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": sampled_paths,
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            sample = {
                "prompt": prompt,
                "video_meta": video_meta
            }
            samples.append(sample)
    
    return samples


# ==========================================
# DAVIS17 数据集加载
# ==========================================
def load_davis17_dataset(split: str = "train") -> List[Dict]:
    """
    加载 DAVIS17 数据集
    """
    samples = []
    
    # 加载 meta_expressions
    meta_file = os.path.join(DAVIS17_PATH, "meta_expressions", split, "meta_expressions.json")
    if not os.path.exists(meta_file):
        print(f"Warning: {meta_file} not found, skipping DAVIS17")
        return samples
    
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    
    # 加载 mask_dict
    mask_dict_file = os.path.join(DAVIS17_PATH, split, "mask_dict.pkl")
    if not os.path.exists(mask_dict_file):
        print(f"Warning: {mask_dict_file} not found, skipping DAVIS17")
        return samples
    
    with open(mask_dict_file, 'rb') as f:
        mask_dict = pickle.load(f)
    
    # 视频帧根目录
    video_base_dir = os.path.join(DAVIS17_PATH, split, "JPEGImages")
    
    videos = meta_data.get('videos', {})
    
    for video_name, video_info in videos.items():
        expressions = video_info.get('expressions', {})
        frames = video_info.get('frames', [])
        
        # 视频帧目录
        video_dir = os.path.join(video_base_dir, video_name)
        if not os.path.exists(video_dir):
            continue
        
        # 获取所有帧路径
        frame_paths = []
        for frame_name in frames:
            frame_path = os.path.join(video_dir, f"{frame_name}.jpg")
            if os.path.exists(frame_path):
                frame_paths.append(frame_path)
        
        if not frame_paths:
            continue
        
        # 获取视频尺寸
        try:
            with Image.open(frame_paths[0]) as img:
                width, height = img.size
        except:
            height, width = 480, 854
        
        for exp_id, exp_info in expressions.items():
            expression = exp_info.get('exp', '')
            obj_id = exp_info.get('obj_id', '')
            
            # 构建 anno_id (DAVIS 格式通常是 video_name_obj_id)
            anno_ids = [f"{video_name}_{obj_id}"]
            
            # 使用混合采样策略
            sampled_paths, sampled_indices = get_hybrid_sampled_frames(
                frame_paths, anno_ids, mask_dict, max_frames=64
            )
            
            # 计算采样时间 (DAVIS 通常是 480p, 假设 fps=24)
            sampled_times = compute_sampled_times(sampled_indices, fps=24.0)
            
            # 构建 video_meta
            video_meta = {
                "video": video_name,
                "fps": 24.0,
                "height": height,
                "width": width,
                "target_anno_ids": anno_ids,
                "sampled_frame_ids": sampled_indices,
                "sampled_times": sampled_times,
                "sampled_frame_paths": sampled_paths
            }
            
            # 构建 prompt
            prompt_text = PROMPT_TEMPLATE.format(question=f"Locate the {expression}.")
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": sampled_paths,
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            sample = {
                "prompt": prompt,
                "video_meta": video_meta
            }
            samples.append(sample)
    
    return samples


# ==========================================
# MeViS 数据集加载
# ==========================================
def load_mevis_dataset(split: str = "train") -> List[Dict]:
    """
    加载 MeViS 数据集
    """
    samples = []
    
    # 加载 meta_expressions
    meta_file = os.path.join(MEVIS_PATH, split, "meta_expressions.json")
    if not os.path.exists(meta_file):
        print(f"Warning: {meta_file} not found, skipping MeViS")
        return samples
    
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    
    # 加载 mask_dict
    mask_dict_file = os.path.join(MEVIS_PATH, split, "mask_dict.json")
    if not os.path.exists(mask_dict_file):
        print(f"Warning: {mask_dict_file} not found, skipping MeViS")
        return samples
    
    with open(mask_dict_file, 'r') as f:
        mask_dict = json.load(f)
    
    # 视频帧根目录
    video_base_dir = os.path.join(MEVIS_PATH, split, "JPEGImages")
    
    videos = meta_data.get('videos', {})
    
    for video_name, video_info in videos.items():
        expressions = video_info.get('expressions', {})
        
        # 视频帧目录
        video_dir = os.path.join(video_base_dir, video_name)
        if not os.path.exists(video_dir):
            continue
        
        # 获取所有帧路径
        frame_paths = get_frame_paths_from_video_dir(video_dir)
        if not frame_paths:
            continue
        
        # 获取视频尺寸
        try:
            with Image.open(frame_paths[0]) as img:
                width, height = img.size
        except:
            height, width = 720, 1280
        
        for exp_id, exp_info in expressions.items():
            expression = exp_info.get('exp', '')
            anno_ids = exp_info.get('anno_id', [])
            
            if not isinstance(anno_ids, list):
                anno_ids = [anno_ids]
            
            # 转换为字符串格式的 anno_id
            target_anno_ids = [str(aid) for aid in anno_ids]
            
            # 使用混合采样策略
            sampled_paths, sampled_indices = get_hybrid_sampled_frames(
                frame_paths, target_anno_ids, mask_dict, max_frames=64
            )
            
            # 计算采样时间 (假设 fps=5)
            sampled_times = compute_sampled_times(sampled_indices, fps=5.0)
            
            # 构建 video_meta
            video_meta = {
                "video": video_name,
                "fps": 5.0,
                "height": height,
                "width": width,
                "target_anno_ids": target_anno_ids,
                "sampled_frame_ids": sampled_indices,
                "sampled_times": sampled_times,
                "sampled_frame_paths": sampled_paths
            }
            
            # 构建 prompt
            prompt_text = PROMPT_TEMPLATE.format(question=f"Locate the {expression}.")
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": sampled_paths,
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            sample = {
                "prompt": prompt,
                "video_meta": video_meta
            }
            samples.append(sample)
    
    return samples


# ==========================================
# RVOS 数据集加载
# ==========================================
def load_rvos_dataset(split: str = "train") -> List[Dict]:
    """
    加载 RVOS 数据集
    """
    samples = []
    
    # 加载 meta_expressions
    meta_file = os.path.join(RVOS_PATH, "meta_expressions", split, "meta_expressions.json")
    if not os.path.exists(meta_file):
        print(f"Warning: {meta_file} not found, skipping RVOS")
        return samples
    
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    
    # 加载 mask_dict
    mask_dict_file = os.path.join(RVOS_PATH, "mask_dict.pkl")
    if not os.path.exists(mask_dict_file):
        print(f"Warning: {mask_dict_file} not found, skipping RVOS")
        return samples
    
    with open(mask_dict_file, 'rb') as f:
        mask_dict = pickle.load(f)
    
    # 视频帧根目录
    video_base_dir = os.path.join(RVOS_PATH, split, "JPEGImages")
    
    videos = meta_data.get('videos', {})
    
    for video_name, video_info in videos.items():
        expressions = video_info.get('expressions', {})
        frames = video_info.get('frames', [])
        
        # 视频帧目录
        video_dir = os.path.join(video_base_dir, video_name)
        if not os.path.exists(video_dir):
            continue
        
        # 直接构建帧路径列表（不检查文件是否存在以提高速度）
        frame_paths = [os.path.join(video_dir, f"{frame_name}.jpg") for frame_name in frames]
        
        if not frame_paths:
            continue
        
        # 获取视频尺寸（只检查第一帧）
        width, height = 720, 1280
        if frame_paths:
            try:
                first_frame = next((p for p in frame_paths if os.path.exists(p)), None)
                if first_frame:
                    with Image.open(first_frame) as img:
                        width, height = img.size
            except:
                pass
        
        for exp_id, exp_info in expressions.items():
            expression = exp_info.get('exp', '')
            obj_id = exp_info.get('obj_id', '')
            
            # 构建 anno_id
            anno_ids = [f"{video_name}_{obj_id}"]
            
            # 使用混合采样策略
            sampled_paths, sampled_indices = get_hybrid_sampled_frames(
                frame_paths, anno_ids, mask_dict, max_frames=64
            )
            
            # 计算采样时间 (假设 fps=5)
            sampled_times = compute_sampled_times(sampled_indices, fps=5.0)
            
            # 构建 video_meta
            video_meta = {
                "video": video_name,
                "fps": 5.0,
                "height": height,
                "width": width,
                "target_anno_ids": anno_ids,
                "sampled_frame_ids": sampled_indices,
                "sampled_times": sampled_times,
                "sampled_frame_paths": sampled_paths
            }
            
            # 构建 prompt
            prompt_text = PROMPT_TEMPLATE.format(question=f"Locate the {expression}.")
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": sampled_paths,
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            sample = {
                "prompt": prompt,
                "video_meta": video_meta
            }
            samples.append(sample)
    
    return samples


def load_ref_sav_dataset() -> List[Dict]:
    """
    加载 Ref-SAV 数据集
    """
    samples = []
    
    # 加载 Ref-SAV.json
    meta_file = os.path.join(REF_SAV_PATH, "Ref-SAV.json")
    if not os.path.exists(meta_file):
        print(f"Warning: {meta_file} not found, skipping Ref-SAV")
        return samples
    
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    
    # Ref-SAV 数据格式特殊，需要解析
    for video_id, video_info in meta_data.items():
        video_path = video_info.get('video_path', '')
        objects = video_info.get('objects', {})
        
        # 获取视频帧 (Ref-SAV 是视频文件，需要提取帧)
        # 这里假设帧已经提取到特定目录
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.join(REF_SAV_PATH, "frames", video_name)
        
        if not os.path.exists(video_dir):
            # 如果帧目录不存在，跳过
            continue
        
        # 获取所有帧路径
        frame_paths = get_frame_paths_from_video_dir(video_dir)
        if not frame_paths:
            continue
        
        # 采样帧
        sampled_paths, sampled_indices = sample_frames_uniform(frame_paths, max_frames=64)
        
        # 获取视频尺寸
        try:
            with Image.open(frame_paths[0]) as img:
                width, height = img.size
        except:
            height, width = 720, 1280
        
        # 计算采样时间 (假设 fps=5)
        sampled_times = compute_sampled_times(sampled_indices, fps=5.0)
        
        for obj_id, obj_info in objects.items():
            expression = obj_info.get('video_caption', obj_info.get('image_caption', ''))
            anno_id = obj_info.get('obj_id', obj_id)
            
            # 构建 video_meta
            video_meta = {
                "video": video_name,
                "fps": 5.0,
                "height": height,
                "width": width,
                "target_anno_ids": [str(anno_id)],
                "sampled_frame_ids": sampled_indices,
                "sampled_times": sampled_times,
                "sampled_frame_paths": sampled_paths
            }
            
            # 构建 prompt
            prompt_text = PROMPT_TEMPLATE.format(question=f"Locate the {expression}.")
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": sampled_paths,
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            sample = {
                "prompt": prompt,
                "video_meta": video_meta
            }
            samples.append(sample)
    
    return samples


# ==========================================
# 主加载函数
# ==========================================
def load_all_datasets(
    load_ref_seg: bool = True,
    load_video_datas: bool = True,
    total_samples: Optional[int] = None,  # 替换原来的 max_samples_per_dataset
    seed: int = 42 # 固定随机种子，保证每次抽样的桶是可复现的
) -> DatasetDict:
    """
    加载所有数据集，支持指定总样本量并从各子数据集中均匀随机抽样
    """
    import random
    random.seed(seed)
    
    # 用字典暂存各个子数据集
    datasets_dict = {}
    
    # ========== ref_seg 数据集 ==========
    if load_ref_seg:
        print("Loading RefCOCO...")
        datasets_dict['refcoco'] = load_ref_dataset(REFCOCO_PATH, "train", "refcoco")
        
        print("Loading RefCOCO+...")
        datasets_dict['refcoco+'] = load_ref_dataset(REFCOCO_PLUS_PATH, "train", "refcoco+")
        
        print("Loading RefCOCOg...")
        datasets_dict['refcocog'] = load_ref_dataset(REFCOCOG_PATH, "train", "refcocog")
    
    # ========== video_datas 数据集 ==========
    if load_video_datas:
        print("Loading ReVOS...")
        datasets_dict['revos'] = load_revos_dataset("train")
        
        print("Loading DAVIS17...")
        datasets_dict['davis17'] = load_davis17_dataset("train")
        
        print("Loading MeViS...")
        datasets_dict['mevis'] = load_mevis_dataset("train")
        
        print("Loading RVOS...")
        datasets_dict['rvos'] = load_rvos_dataset("train")
        
        print("Loading Ref-SAV...")
        datasets_dict['ref_sav'] = load_ref_sav_dataset()
    
    # 过滤掉加载失败或为空的数据集
    datasets_dict = {name: data for name, data in datasets_dict.items() if len(data) > 0}
    
    all_train_samples = []
    
    # ========== 核心逻辑：均匀分桶抽样 ==========
    if total_samples is not None and len(datasets_dict) > 0:
        num_datasets = len(datasets_dict)
        # 计算每个数据集的抽样配额
        quota_per_dataset = total_samples // num_datasets
        
        print(f"\n[Sampling] Total target: {total_samples}, Datasets: {num_datasets}, Quota per dataset: {quota_per_dataset}")
        
        for name, samples in datasets_dict.items():
            # 如果某个数据集自身数量不够配额，就全拿；够的话就随机抽样
            take_num = min(len(samples), quota_per_dataset)
            sampled_data = random.sample(samples, take_num)
            all_train_samples.extend(sampled_data)
            print(f"  -> Sampled {take_num} from {name} (Original size: {len(samples)})")
    else:
        # 不限制数量，全量加载
        for name, samples in datasets_dict.items():
            all_train_samples.extend(samples)
            print(f"  -> Loaded all {len(samples)} from {name}")
    
    # 抽样合并后，必须全局打乱！防止模型连续学习同一个数据集产生分布偏移
    print("\n[Shuffling] Shuffling the aggregated dataset...")
    random.shuffle(all_train_samples)
    
    print(f"Total train samples ready for training: {len(all_train_samples)}")
    
    # 创建 HuggingFace Dataset
    train_dataset = Dataset.from_list(all_train_samples)
    return DatasetDict({"train": train_dataset})


def create_and_save_dataset(
    output_path: str = "./hf_dataset",
    load_ref_seg: bool = True,
    load_video_datas: bool = True,
    max_samples_per_dataset: Optional[int] = None
):
    """
    创建并保存数据集到磁盘
    """
    try:
        dataset = load_all_datasets(
            load_ref_seg=load_ref_seg,
            load_video_datas=load_video_datas,
            max_samples_per_dataset=max_samples_per_dataset
        )
        
        print(f"\nDataset info:")
        print(f"  Train samples: {len(dataset['train'])}")
        
        print(f"\nSaving dataset to {output_path}...")
        
        os.makedirs(output_path, exist_ok=True)
        
        dataset.save_to_disk(output_path)
        print("Dataset saved successfully!")
        
        return dataset
    except Exception as e:
        print(f"Error saving dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    dataset = create_and_save_dataset(
        output_path="/home/ma-user/modelarts/user-job-dir/src/negpoint/hf_dataset",
        load_ref_seg=False,
        load_video_datas=True,
        max_samples_per_dataset=None
    )
    
    if dataset is not None:
        print("\nSample example:")
        sample = dataset["train"][0]
        print(f"Prompt: {sample['prompt']}")
        print(f"Video meta: {sample['video_meta']}")
    else:
        print("\nFailed to create or save dataset.")
