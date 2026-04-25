import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from pycocotools import mask as mask_utils


# ==========================================
# 全局 Mask 缓存管理
# ==========================================
class MaskManager:
    """
    统一管理不同数据集的 mask 数据
    支持延迟加载和缓存
    """
    def __init__(self):
        self._mask_caches = {}
        self._mask_paths = {}
        self._annotation_paths = {}
    
    def register_dataset(self, dataset_name: str, mask_path: str = None, 
                         mask_type: str = "auto", annotation_path: str = None):
        """
        注册数据集的 mask 文件
        
        Args:
            dataset_name: 数据集名称 (如 "revos", "davis17", "mevis" 等)
            mask_path: mask 文件路径 (可选)
            mask_type: mask 文件类型 ("json", "pkl", "png", "auto")
            annotation_path: 标注文件路径 (用于某些数据集)
        """
        self._mask_paths[dataset_name] = {
            "path": mask_path,
            "type": mask_type,
            "loaded": False
        }
        if annotation_path:
            self._annotation_paths[dataset_name] = annotation_path
    
    def _load_mask_dict(self, dataset_name: str) -> Optional[Dict]:
        """加载指定数据集的 mask 字典"""
        if dataset_name not in self._mask_paths:
            return None
        
        info = self._mask_paths[dataset_name]
        
        # 如果已经加载到缓存，直接返回
        if info["loaded"] and dataset_name in self._mask_caches:
            return self._mask_caches[dataset_name]
        
        mask_path = info["path"]
        if mask_path is None or not os.path.exists(mask_path):
            # DAVIS17 和 Ref-SAV 可能没有 mask_dict，返回空字典
            self._mask_caches[dataset_name] = {}
            self._mask_paths[dataset_name]["loaded"] = True
            return {}
        
        # 自动检测文件类型
        mask_type = info["type"]
        if mask_type == "auto":
            if mask_path.endswith('.json'):
                mask_type = "json"
            elif mask_path.endswith('.pkl') or mask_path.endswith('.pickle'):
                mask_type = "pkl"
        
        try:
            if mask_type == "json":
                with open(mask_path, 'r') as f:
                    mask_dict = json.load(f)
            elif mask_type == "pkl":
                with open(mask_path, 'rb') as f:
                    mask_dict = pickle.load(f)
            else:
                print(f"Warning: Unknown mask type: {mask_type}")
                return None
            
            self._mask_caches[dataset_name] = mask_dict
            self._mask_paths[dataset_name]["loaded"] = True
            return mask_dict
            
        except Exception as e:
            print(f"Error loading mask file {mask_path}: {e}")
            return None
    
    def get_mask(self, dataset_name: str, anno_id: str, frame_idx: int, 
                 video_name: str = None) -> Optional[np.ndarray]:
        """
        获取指定数据集、标注ID和帧索引的 mask
        
        Args:
            dataset_name: 数据集名称
            anno_id: 标注ID
            frame_idx: 帧索引
            video_name: 视频名称 (用于 DAVIS17 和 Ref-SAV)
        
        Returns:
            binary mask 数组或 None
        """
        # DAVIS17: 从 PNG 文件加载
        if dataset_name == "davis17" and video_name:
            return self._load_davis17_mask(video_name, anno_id, frame_idx)
        
        # Ref-SAV: 从 JSON 文件加载
        if dataset_name == "ref_sav" and video_name:
            return self._load_ref_sav_mask(video_name, anno_id, frame_idx)
        
        # 其他数据集: 从 mask_dict 加载
        mask_dict = self._load_mask_dict(dataset_name)
        if mask_dict is None:
            return None
        
        anno_id_str = str(anno_id)
        if anno_id_str not in mask_dict:
            return None
        
        mask_list = mask_dict[anno_id_str]
        
        if not isinstance(mask_list, list):
            return None
        
        if frame_idx < 0 or frame_idx >= len(mask_list):
            return None
        
        rle = mask_list[frame_idx]
        if rle is None:
            return None
        
        # 处理 RLE 格式
        try:
            if isinstance(rle, dict):
                # 确保 counts 是 bytes 类型
                if isinstance(rle.get('counts'), str):
                    rle = rle.copy()
                    rle['counts'] = rle['counts'].encode('utf-8')
                return mask_utils.decode(rle)
            else:
                return None
        except Exception as e:
            print(f"Error decoding mask for {dataset_name}/{anno_id}/{frame_idx}: {e}")
            return None
    
    def _load_davis17_mask(self, video_name: str, anno_id: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        从 DAVIS17 的 PNG 文件加载 mask
        
        Args:
            video_name: 视频名称 (如 "bear")
            anno_id: 对象ID (如 "1")
            frame_idx: 帧索引
        """
        base_path = "/home/ma-user/sfs_turbo/qinianwang/datasets/Sa2VA-Training/video_datas/davis17/train/Annotations"
        
        # DAVIS17 的 PNG 文件命名格式: {frame_idx:05d}.png
        mask_path = os.path.join(base_path, video_name, f"{frame_idx:05d}.png")
        
        if not os.path.exists(mask_path):
            return None
        
        try:
            import cv2
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            # DAVIS17 使用 BGR 格式的 PNG 文件，其中蓝色通道包含对象ID
            # 对象1: 蓝色通道 = 128
            # 对象2: 蓝色通道 = 192
            # 对象3: 蓝色通道 = 224
            # ...
            
            # 如果是 3 通道图像，只使用蓝色通道
            if len(mask_img.shape) == 3:
                mask_img = mask_img[:, :, 2]  # 蓝色通道
            
            # DAVIS17 的对象ID从1开始
            obj_id = int(anno_id)
            
            # DAVIS17 的颜色编码:
            # 对象1: 128
            # 对象2: 192
            # 对象3: 224
            # ...
            # 公式: color = 128 + (obj_id - 1) * 64
            color = 128 + (obj_id - 1) * 64
            
            # 创建二进制 mask
            binary_mask = (mask_img == color).astype(np.uint8)
            
            return binary_mask
            
        except Exception as e:
            print(f"Error loading DAVIS17 mask for {video_name}/{anno_id}/{frame_idx}: {e}")
            return None
    
    def _load_ref_sav_mask(self, video_name: str, anno_id: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        从 Ref-SAV 的 JSON 文件加载 mask
        
        Args:
            video_name: 视频名称 (如 "sav_000002")
            anno_id: 对象ID
            frame_idx: 帧索引
        """
        base_path = "/home/ma-user/sfs_turbo/qinianwang/datasets/Sa2VA-Training/video_datas/ref_sav/sam_v_full/sav_train"
        
        # Ref-SAV 的 JSON 文件命名格式: {video_name}_auto.json 或 {video_name}_manual.json
        # 文件位于 sav_000 子目录下
        json_path = os.path.join(base_path, "sav_000", f"{video_name}_auto.json")
        
        if not os.path.exists(json_path):
            json_path = os.path.join(base_path, "sav_000", f"{video_name}_manual.json")
        
        if not os.path.exists(json_path):
            return None
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Ref-SAV 的 JSON 格式:
            # - masklet: 列表的列表，每个子列表是一个对象的 mask 序列
            # - masklet_id: 对象ID列表
            # - masklet_first_appeared_frame: 每个对象首次出现的帧索引
            
            masklets = data.get('masklet', [])
            masklet_ids = data.get('masklet_id', [])
            first_frames = data.get('masklet_first_appeared_frame', [])
            
            # 获取指定对象的 masklet
            obj_id = int(anno_id)
            
            # 找到对应的 masklet 索引
            masklet_idx = None
            for i, mid in enumerate(masklet_ids):
                if mid == obj_id:
                    masklet_idx = i
                    break
            
            if masklet_idx is None or masklet_idx >= len(masklets):
                return None
            
            # 获取该对象的 mask 序列
            mask_sequence = masklets[masklet_idx]
            first_frame = first_frames[masklet_idx] if masklet_idx < len(first_frames) else 0
            
            # 计算相对帧索引
            relative_frame_idx = frame_idx - first_frame
            
            if relative_frame_idx < 0 or relative_frame_idx >= len(mask_sequence):
                return None
            
            mask_data = mask_sequence[relative_frame_idx]
            
            # 解析 mask
            if isinstance(mask_data, dict) and 'size' in mask_data and 'counts' in mask_data:
                # RLE 格式
                if isinstance(mask_data.get('counts'), str):
                    mask_data = mask_data.copy()
                    mask_data['counts'] = mask_data['counts'].encode('utf-8')
                return mask_utils.decode(mask_data)
            elif isinstance(mask_data, np.ndarray):
                return mask_data.astype(np.uint8)
            else:
                return None
                
        except Exception as e:
            print(f"Error loading Ref-SAV mask for {video_name}/{anno_id}/{frame_idx}: {e}")
            return None
    
    def get_mask_list(self, dataset_name: str, anno_id: str) -> Optional[List]:
        """获取指定标注ID的所有帧 mask 列表"""
        mask_dict = self._load_mask_dict(dataset_name)
        if mask_dict is None:
            return None
        
        anno_id_str = str(anno_id)
        if anno_id_str not in mask_dict:
            return None
        
        return mask_dict[anno_id_str]


# 全局 MaskManager 实例
GLOBAL_MASK_MANAGER = MaskManager()


# ==========================================
# 数据集特定的标注处理函数
# ==========================================

class AnnotationHandler:
    """
    处理不同数据集的标注格式差异
    """
    
    @staticmethod
    def get_dataset_from_video_name(video_name: str) -> str:
        """
        根据视频名称推断数据集名称
        """
        video_lower = video_name.lower()
        
        # ReVOS 视频名通常包含子数据集名称，如 "UVO/all/xxx" 或 "LV-VIS/train/xxx"
        if any(prefix in video_name for prefix in ["UVO/", "LV-VIS/", "MOSE/", "OVIS/", "TAO/"]):
            return "revos"
        
        # DAVIS17 视频名通常是简单的英文单词
        if video_lower in ["bear", "bmx-bumps", "boat", "breakdance", "bus", 
                          "car-turn", "cows", "dance-jump", "dog-agility",
                          "drift-chicane", "drift-straight", "goat", "hike",
                          "hockey", "horse-jump", "kite-surf", "lucia", "mallard-fly",
                          "mallard-water", "motocross-bumps", "motocross-jump",
                          "motorbike", "paragliding", "parkour", "planes-water",
                          "rally", "rhino", "rollerblade", "scooter-black",
                          "scooter-gray", "soapbox", "soccerball", "stroller",
                          "surf", "swing", "tennis", "train"]:
            return "davis17"
        
        # MeViS 视频名通常是12位十六进制字符串
        if len(video_name) == 12 and all(c in '0123456789abcdef' for c in video_lower):
            return "mevis"
        
        # RVOS 视频名通常是10位十六进制字符串
        if len(video_name) == 10 and all(c in '0123456789abcdef' for c in video_lower):
            return "rvos"
        
        # Ref-SAV 视频名通常以 "sav_" 开头
        if video_name.startswith("sav_"):
            return "ref_sav"
        
        # ref_seg 数据集 (RefCOCO系列)
        if any(prefix in video_name for prefix in ["refcoco_", "refcoco+", "refcocog_"]):
            return "ref_seg"
        
        # 默认返回 unknown
        return "unknown"
    
    @staticmethod
    def resolve_anno_ids(video_meta: Dict, dataset_name: str) -> List[str]:
        """
        根据数据集类型解析标注ID
        
        不同数据集的 anno_id 格式不同:
        - ReVOS: 直接使用 anno_id (数字)
        - DAVIS17: 直接使用对象ID (数字)
        - RVOS: video_name_obj_id
        - MeViS: 直接使用 anno_id (数字)
        - Ref-SAV: 直接使用对象ID (数字)
        """
        target_anno_ids = video_meta.get("target_anno_ids", [])
        video_name = video_meta.get("video", "")
        
        resolved_ids = []
        for anno_id in target_anno_ids:
            anno_id_str = str(anno_id)
            
            if dataset_name == "rvos":
                # RVOS: 格式为 "video_name_obj_id"
                if not anno_id_str.startswith(video_name):
                    resolved_ids.append(f"{video_name}_{anno_id_str}")
                else:
                    resolved_ids.append(anno_id_str)
            else:
                # 其他数据集: 直接使用 anno_id
                resolved_ids.append(anno_id_str)
        
        return resolved_ids
    
    @staticmethod
    def map_frame_idx(video_meta: Dict, predicted_time: float) -> Tuple[int, str]:
        """
        将预测的时间映射到实际的帧索引和帧路径
        
        Returns:
            (frame_idx, frame_path)
        """
        sampled_times = video_meta.get("sampled_times", [])
        sampled_frame_paths = video_meta.get("sampled_frame_paths", [])
        sampled_frame_ids = video_meta.get("sampled_frame_ids", [])
        
        if not sampled_times or not sampled_frame_paths:
            return 0, ""
        
        # 找到最接近预测时间的采样帧
        closest_idx = int(np.argmin([abs(t - predicted_time) for t in sampled_times]))
        
        # 获取对应的原始帧索引和帧路径
        if closest_idx < len(sampled_frame_ids):
            frame_idx = sampled_frame_ids[closest_idx]
        else:
            frame_idx = closest_idx
            
        if closest_idx < len(sampled_frame_paths):
            frame_path = sampled_frame_paths[closest_idx]
        else:
            frame_path = ""
        
        return frame_idx, frame_path


# ==========================================
# Mask 加载工具函数
# ==========================================

def load_all_masks(base_data_path: str = "/home/ma-user/sfs_turbo/qinianwang/datasets/Sa2VA-Training"):
    """
    加载所有数据集的 mask 文件
    """
    # ReVOS
    revos_mask_path = os.path.join(base_data_path, "video_datas", "revos", "mask_dict.json")
    if os.path.exists(revos_mask_path):
        GLOBAL_MASK_MANAGER.register_dataset("revos", revos_mask_path, "json")
        print(f"Registered ReVOS masks: {revos_mask_path}")
    
    # DAVIS17 (使用 PNG 文件，不注册 mask_dict)
    davis_annotation_path = os.path.join(base_data_path, "video_datas", "davis17", "meta_expressions", "train", "meta_expressions.json")
    GLOBAL_MASK_MANAGER.register_dataset("davis17", mask_path=None, mask_type="png", annotation_path=davis_annotation_path)
    print(f"Registered DAVIS17 masks (PNG format)")
    
    # MeViS
    mevis_mask_path = os.path.join(base_data_path, "video_datas", "mevis", "train", "mask_dict.json")
    if os.path.exists(mevis_mask_path):
        GLOBAL_MASK_MANAGER.register_dataset("mevis", mevis_mask_path, "json")
        print(f"Registered MeViS masks: {mevis_mask_path}")
    
    # RVOS
    rvos_mask_path = os.path.join(base_data_path, "video_datas", "rvos", "mask_dict.pkl")
    if os.path.exists(rvos_mask_path):
        GLOBAL_MASK_MANAGER.register_dataset("rvos", rvos_mask_path, "pkl")
        print(f"Registered RVOS masks: {rvos_mask_path}")
    
    # Ref-SAV (使用 JSON 文件，不注册 mask_dict)
    ref_sav_annotation_path = os.path.join(base_data_path, "video_datas", "ref_sav", "Ref-SAV.json")
    GLOBAL_MASK_MANAGER.register_dataset("ref_sav", mask_path=None, mask_type="json", annotation_path=ref_sav_annotation_path)
    print(f"Registered Ref-SAV masks (JSON format)")


def get_instance_masks_from_meta(video_meta: Dict, frame_idx: int) -> List[np.ndarray]:
    """
    根据 video_meta 获取指定帧的所有实例 mask
    
    Args:
        video_meta: 视频元数据
        frame_idx: 帧索引 (原始帧索引，不是采样后的索引)
    
    Returns:
        List of binary masks
    """
    video_name = video_meta.get("video", "")
    target_anno_ids = video_meta.get("target_anno_ids", [])
    
    # 确定数据集名称
    dataset_name = AnnotationHandler.get_dataset_from_video_name(video_name)
    
    if dataset_name == "unknown":
        return []
    
    # 解析标注ID
    resolved_anno_ids = AnnotationHandler.resolve_anno_ids(video_meta, dataset_name)
    
    # 获取 masks
    masks = []
    for anno_id in resolved_anno_ids:
        mask = GLOBAL_MASK_MANAGER.get_mask(dataset_name, anno_id, frame_idx, video_name)
        if mask is not None:
            masks.append(mask.astype(np.uint8))
    
    return masks


def get_masks_for_predicted_time(video_meta: Dict, predicted_time: float) -> Tuple[List[np.ndarray], str]:
    """
    根据预测的时间获取对应的 masks 和帧路径
    
    Args:
        video_meta: 视频元数据
        predicted_time: 预测的时间（秒）
    
    Returns:
        (masks, frame_path)
    """
    # 将预测时间映射到帧索引和帧路径
    frame_idx, frame_path = AnnotationHandler.map_frame_idx(video_meta, predicted_time)
    
    # 获取 masks
    masks = get_instance_masks_from_meta(video_meta, frame_idx)
    
    return masks, frame_path


# ==========================================
# 坐标转换工具
# ==========================================

def scale_point(pt: List[float], w: int, h: int) -> Tuple[int, int]:
    """
    归一化坐标(0-1000)转像素坐标
    
    Args:
        pt: [x, y] 归一化坐标 (0-1000)
        w: 图像宽度
        h: 图像高度
    
    Returns:
        (px, py) 像素坐标
    """
    return (
        min(max(int(pt[0] * w / 1000), 0), w - 1),
        min(max(int(pt[1] * h / 1000), 0), h - 1)
    )


def normalize_point(pt: Tuple[int, int], w: int, h: int) -> List[float]:
    """
    像素坐标转归一化坐标(0-1000)
    
    Args:
        pt: (px, py) 像素坐标
        w: 图像宽度
        h: 图像高度
    
    Returns:
        [x, y] 归一化坐标 (0-1000)
    """
    px, py = pt
    return [
        min(max(px * 1000 / w, 0), 1000),
        min(max(py * 1000 / h, 0), 1000)
    ]


# ==========================================
# 可视化工具 (用于调试)
# ==========================================

def visualize_mask_on_image(image_path: str, masks: List[np.ndarray], 
                            points_pos: Optional[List[Tuple[int, int]]] = None,
                            point_neg: Optional[Tuple[int, int]] = None,
                            output_path: Optional[str] = None):
    """
    在图像上可视化 mask 和点 (用于调试)
    """
    import cv2
    
    img = cv2.imread(image_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    # 绘制 masks
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        mask_bool = mask.astype(bool)
        img[mask_bool] = img[mask_bool] * 0.5 + np.array(color) * 0.5
    
    # 绘制正样本点
    if points_pos:
        for pt in points_pos:
            cv2.circle(img, pt, 5, (0, 255, 0), -1)
            cv2.circle(img, pt, 7, (255, 255, 255), 2)
    
    # 绘制负样本点
    if point_neg:
        cv2.circle(img, point_neg, 5, (0, 0, 255), -1)
        cv2.circle(img, point_neg, 7, (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img


# ==========================================
# 测试代码
# ==========================================

if __name__ == "__main__":
    # 测试加载所有 masks
    load_all_masks()
    
    # 测试获取 mask
    print("\nTesting mask retrieval...")
    
    # 测试 DAVIS17
    test_meta_davis = {
        "video": "bear",
        "target_anno_ids": ["1"],
        "sampled_times": [0.0, 1.0, 2.0],
        "sampled_frame_ids": [0, 24, 48],
        "sampled_frame_paths": ["path/to/frame0.jpg", "path/to/frame24.jpg", "path/to/frame48.jpg"]
    }
    
    dataset = AnnotationHandler.get_dataset_from_video_name("bear")
    print(f"Dataset for 'bear': {dataset}")
    
    resolved_ids = AnnotationHandler.resolve_anno_ids(test_meta_davis, dataset)
    print(f"Resolved anno IDs: {resolved_ids}")
    
    # 测试时间映射
    frame_idx, frame_path = AnnotationHandler.map_frame_idx(test_meta_davis, 1.5)
    print(f"Time 1.5s -> frame_idx: {frame_idx}, path: {frame_path}")
    
    # 测试获取 mask
    masks = get_instance_masks_from_meta(test_meta_davis, 0)
    print(f"Masks for frame 0: {len(masks)}")
    if masks:
        print(f"First mask shape: {masks[0].shape}")
