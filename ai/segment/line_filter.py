"""
行图片过滤器
根据行高度等条件过滤行图片

使用方法：
    from ai.segment.line_filter import filter_lines_by_height, LineHeightFilter
    
    # 方式1：函数式过滤
    valid_lines = filter_lines_by_height("line_dir", min_h=30, max_h=100)
    
    # 方式2：类方式过滤（推荐）
    filter = LineHeightFilter(min_h=30, max_h=100)
    valid, invalid = filter.filter_with_report(line_paths)
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image


def get_line_height(img_path: str) -> Optional[int]:
    """
    获取行图片的高度（像素）
    
    Args:
        img_path: 行图片路径
    
    Returns:
        图片高度（像素），获取失败返回None
    """
    try:
        with Image.open(img_path) as img:
            return img.height
    except Exception:
        return None


def filter_lines_by_height(
    line_dir: str,
    min_h: int = 30,
    max_h: int = 100,
    recursive: bool = False
) -> List[str]:
    """
    按行高度过滤行图片（函数式接口）
    
    Args:
        line_dir: 行图片所在目录
        min_h: 最小高度阈值（像素），低于此值的行会被过滤掉
        max_h: 最大高度阈值（像素），高于此值的行会被过滤掉
        recursive: 是否递归搜索子目录，默认False
    
    Returns:
        符合条件的行图片路径列表
    """
    valid_lines = []
    
    line_path = Path(line_dir)
    if not line_path.exists():
        return []
    
    # 获取所有png图片
    pattern = "**/*.png" if recursive else "*.png"
    img_files = list(line_path.glob(pattern))
    
    for img_file in img_files:
        height = get_line_height(str(img_file))
        if height is None:
            continue
            
        # 判断是否在指定高度范围内
        if min_h <= height <= max_h:
            valid_lines.append(str(img_file))
    
    return valid_lines


def filter_lines_by_height_with_report(
    line_dir: str,
    min_h: int = 30,
    max_h: int = 100,
    recursive: bool = False
) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    按行高度过滤行图片（带详细报告）
    
    Args:
        line_dir: 行图片所在目录
        min_h: 最小高度阈值（像素）
        max_h: 最大高度阈值（像素）
        recursive: 是否递归搜索子目录
    
    Returns:
        (valid_lines, invalid_lines_with_height)
        - valid_lines: 符合条件的行图片路径列表
        - invalid_lines_with_height: 不符合条件及原因：(路径, 实际高度)
    """
    valid_lines = []
    invalid_lines = []
    
    line_path = Path(line_dir)
    if not line_path.exists():
        return [], []
    
    pattern = "**/*.png" if recursive else "*.png"
    img_files = list(line_path.glob(pattern))
    
    for img_file in img_files:
        height = get_line_height(str(img_file))
        if height is None:
            continue
            
        if min_h <= height <= max_h:
            valid_lines.append(str(img_file))
        else:
            invalid_lines.append((str(img_file), height))
    
    return valid_lines, invalid_lines


class LineHeightFilter:
    """
    行高度过滤器类
    
    用于过滤掉过高或过矮的行图片，这些行可能是噪声或无效数据
    
    Attributes:
        min_h: 最小高度阈值（像素）
        max_h: 最大高度阈值（像素）
    """
    
    def __init__(self, min_h: int = 30, max_h: int = 100):
        """
        初始化行高度过滤器
        
        Args:
            min_h: 最小高度阈值（像素），默认30
            max_h: 最大高度阈值（像素），默认100
        """
        self.min_h = min_h
        self.max_h = max_h
    
    def is_valid(self, img_path: str) -> bool:
        """
        检查单张行图片是否符合高度要求
        
        Args:
            img_path: 行图片路径
        
        Returns:
            True=符合要求，False=不符合要求
        """
        height = get_line_height(img_path)
        if height is None:
            return False
        return self.min_h <= height <= self.max_h
    
    def filter(self, line_paths: List[str]) -> List[str]:
        """
        批量过滤行图片
        
        Args:
            line_paths: 行图片路径列表
        
        Returns:
            符合条件的行图片路径列表
        """
        return [p for p in line_paths if self.is_valid(p)]
    
    def filter_with_report(self, line_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        批量过滤行图片（带详细报告）
        
        Args:
            line_paths: 行图片路径列表
        
        Returns:
            (valid_lines, invalid_lines)
            - valid_lines: 符合条件的行图片路径列表
            - invalid_lines: 不符合条件 的行图片路径列表
        """
        valid = []
        invalid = []
        for p in line_paths:
            if self.is_valid(p):
                valid.append(p)
            else:
                invalid.append(p)
        return valid, invalid
    
    def get_height(self, img_path: str) -> Optional[int]:
        """获取图片高度"""
        return get_line_height(img_path)
    
    def get_stats(self, line_paths: List[str]) -> dict:
        """
        获取行图片高度统计信息
        
        Args:
            line_paths: 行图片路径列表
        
        Returns:
            统计信息字典，包含：
            - total: 总数
            - valid: 有效数量
            - invalid: 无效数量
            - min_height: 最小高度
            - max_height: 最大高度
            - avg_height: 平均高度
        """
        heights = []
        for p in line_paths:
            h = get_line_height(p)
            if h is not None:
                heights.append(h)
        
        if not heights:
            return {"total": 0, "valid": 0, "invalid": 0}
        
        valid_count = sum(1 for h in heights if self.min_h <= h <= self.max_h)
        
        return {
            "total": len(heights),
            "valid": valid_count,
            "invalid": len(heights) - valid_count,
            "min_height": min(heights),
            "max_height": max(heights),
            "avg_height": sum(heights) / len(heights)
        }