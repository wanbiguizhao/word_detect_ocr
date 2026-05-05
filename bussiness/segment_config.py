from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from pathlib import Path

@dataclass
class Line2CharConfig:
    """行转汉字配置类（支持规则、模型、融合三种执行模式）"""
    
    # ===== 输出目录配置 =====
    output_dir: str = "fusion_results"  # 输出根目录：融合结果保存路径
    image_dir: str = "raw_images"  # 原始图片目录
    rule_json_dir: str = "rule_jsons"  # 规则JSON输出目录
    model_json_dir: str = "model_jsons"  # 模型JSON输出目录
    output_json_dir: str = "fusion_results/json"  # 融合结果JSON输出目录
    output_img_dir: str = "fusion_results/visual"  # 可视化图片输出目录
    save_visual: bool = True  # 是否保存可视化结果

    # ===== 规则执行参数 =====
    rule_narrow_blank_threshold: int = 15  # 窄空白阈值：仅当空白宽度≤此值时才合并
    rule_min_char_width: int = 30  # 最小字符宽度：触发合并的字符宽度阈值

    # ===== 模型执行参数 =====
    model_prob_threshold: float = 0.65  # 模型概率阈值：判断是否为有效汉字
    model_crop_width: int = 101  # 模型切割宽度：单字切割框宽度
    model_target_height: int = 55  # 模型目标高度：单字切割框高度
    
    # ===== 模型路径配置 =====
    model_ae_path: str = ""  # 预训练自编码器权重路径（使用时必须设置）
    model_segment_path: str = ""  # 预训练分割模型权重路径（使用时必须设置）
    model_device: str = "auto"  # 运行设备：auto=自动选择, cuda, cpu

    # ===== 融合执行参数 =====
    fusion_enabled: bool = True  # 是否启用融合模式：True=规则+模型融合

    # ===== 可视化配置 =====
    sep_line_color: Tuple[int, int, int] = (128, 0, 128)  # 分隔线颜色：紫色
    sep_line_width: int = 2  # 分隔线宽度：2像素
    cut_line_width: int = 1  # 切割线宽度：1像素
    start_color: Tuple[int, int, int] = (255, 0, 0)  # 开始位置颜色：红色
    end_color: Tuple[int, int, int] = (0, 255, 0)  # 结束位置颜色：绿色

    # ===== 行高度过滤参数 =====
    min_line_height: int = 30  # 最小行高度：低于此值的行会被过滤
    max_line_height: int = 55  # 最大行高度：高于此值的行会被过滤

    # ===== V2数据结构输出 =====
    v2_output_enabled: bool = True  # 是否启用V2格式输出：生成dataset/v2/血缘索引

    def validate(self) -> None:
        """验证配置参数"""
        assert self.rule_narrow_blank_threshold >= 0, "窄空白阈值不能为负数"
        assert self.rule_min_char_width >= 1, "最小字符宽度至少为1"
        assert 0 <= self.model_prob_threshold <= 1, "模型概率阈值需在0-1之间"
        assert self.min_line_height >= 0, "最小行高度不能为负数"
        assert self.max_line_height >= self.min_line_height, "最大行高度必须大于最小行高度"
        assert self.sep_line_width >= 1, "分隔线宽度至少为1"
        assert self.cut_line_width >= 1, "切割线宽度至少为1"

    def get_rule_config(self) -> dict:
        """获取规则配置字典"""
        return {
            "narrow_blank_threshold": self.rule_narrow_blank_threshold,
            "min_char_width": self.rule_min_char_width
        }

    def get_model_config(self) -> dict:
        """获取模型配置字典"""
        return {
            "prob_threshold": self.model_prob_threshold,
            "crop_width": self.model_crop_width,
            "target_height": self.model_target_height,
            "ae_path": self.model_ae_path,
            "segment_path": self.model_segment_path,
            "device": self.model_device
        }
    
    def get_visual_config(self) -> dict:
        """获取可视化配置字典"""
        return {
            "sep_line_color": self.sep_line_color,
            "sep_line_width": self.sep_line_width,
            "cut_line_width": self.cut_line_width,
            "start_color": self.start_color,
            "end_color": self.end_color
        }
    
    def get_model_paths(self) -> Tuple[str, str]:
        """获取模型路径元组"""
        return (self.model_ae_path, self.model_segment_path)
    
    def get_device(self) -> str:
        """获取运行设备"""
        if self.model_device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.model_device
