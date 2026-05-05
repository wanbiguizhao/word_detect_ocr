"""
主动学习管理器 - 支持多数据集
用于对指定数据集执行主动学习，生成 top_annotate_samples.json

功能：
    1. 支持指定数据集ID（如 pdf01、pdf5823）
    2. 自动查找数据集目录下的规则切割和模型切割结果
    3. 计算主动学习评分（不一致性、不确定性、异常性）
    4. 生成 top_annotate_samples.json 文件
    5. 支持可视化对比

使用方法：
    from bussiness.active_learning_manager import ActiveLearningManager
    
    # 创建管理器（指定数据集）
    manager = ActiveLearningManager(dataset_id="pdf5823")
    
    # 执行主动学习
    top_samples = manager.run(top_n=100)
    
    # 生成可视化对比图（可选）
    manager.generate_visualization(top_samples)
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

# 延迟导入 PIL（仅在需要可视化时导入）
Image = None
ImageDraw = None

def _import_pil():
    global Image, ImageDraw
    if Image is None:
        from PIL import Image as PILImage, ImageDraw as PILImageDraw
        Image = PILImage
        ImageDraw = PILImageDraw


# ===================== 配置类 =====================
@dataclass
class ActiveLearningConfig:
    """主动学习配置"""
    normal_char_ratio: tuple = (0.5, 1.8)
    border_bias_thresh: int = 5
    valley_depth_thresh: float = 0.3
    score_weights: dict = None
    
    def __post_init__(self):
        if self.score_weights is None:
            self.score_weights = {
                "disagree": 0.4,
                "uncertain": 0.3,
                "abnormal": 0.3
            }


# ===================== 核心工具函数 =====================
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def count_disagree_score(rule_boxes, model_boxes):
    num_diff = abs(len(rule_boxes) - len(model_boxes))
    num_score = min(num_diff / 5, 1.0)
    iou_scores = []
    border_bias = []
    for r_box in rule_boxes:
        best_iou = max([calculate_iou(r_box, m_box) for m_box in model_boxes], default=0)
        iou_scores.append(1 - best_iou)
        for m_box in model_boxes:
            bias = abs(r_box[0] - m_box[0]) + abs(r_box[2] - m_box[2])
            border_bias.append(1 if bias > BORDER_BIAS_THRESH else 0)
    iou_score = np.mean(iou_scores) if iou_scores else 1.0
    border_score = np.mean(border_bias) if border_bias else 0.0
    disagree_score = (num_score * 0.4 + iou_score * 0.4 + border_score * 0.2)
    return round(disagree_score, 4)


def uncertain_score(model_pred_probs):
    if not model_pred_probs:
        return 1.0
    entropy = [-p * np.log(p + 1e-8) - (1-p) * np.log(1-p + 1e-8) for p in model_pred_probs]
    return round(np.mean(entropy) / np.log(2), 4)


def abnormal_score(rule_boxes, projection_valley_depth):
    ratio_abnormal = 0
    for box in rule_boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        ratio = w / h if h > 0 else 0
        if not (NORMAL_CHAR_RATIO[0] <= ratio <= NORMAL_CHAR_RATIO[1]):
            ratio_abnormal += 1
    ratio_score = ratio_abnormal / len(rule_boxes) if rule_boxes else 1.0
    valley_score = 1.0 if projection_valley_depth < VALLEY_DEPTH_THRESH else 0.0
    abnormal_score = (ratio_score * 0.6 + valley_score * 0.4)
    return round(abnormal_score, 4)


# ===================== 全局配置 =====================
NORMAL_CHAR_RATIO = (0.5, 1.8)
BORDER_BIAS_THRESH = 5
VALLEY_DEPTH_THRESH = 0.3


# ===================== 主动学习管理器 =====================
class ActiveLearningManager:
    """
    主动学习管理器
    
    支持对指定数据集执行主动学习，生成 top_annotate_samples.json
    
    Attributes:
        dataset_id: 数据集ID（如 pdf01、pdf5823）
        dataset_dir: 数据集目录路径
        config: 主动学习配置
        datahome_dir: 数据根目录
    """
    
    def __init__(self, dataset_id: str, config: ActiveLearningConfig = None):
        """
        初始化主动学习管理器
        
        Args:
            dataset_id: 数据集ID
            config: 主动学习配置（可选）
        """
        self.dataset_id = dataset_id
        self.config = config or ActiveLearningConfig()
        
        # 计算路径
        self.datahome_dir = Path(__file__).resolve().parent / "datahome"
        self.dataset_dir = self.datahome_dir / dataset_id
        
        # 数据集内的子目录
        self.pdf_lines_dir = self.dataset_dir / "pdf_lines"
        
        # 自动检测规则切割目录（优先选择有文件的目录）
        rule_jsons_path = self.dataset_dir / "rule_jsons"
        rule_infer_path = self.dataset_dir / "rule_infer"
        
        if rule_jsons_path.exists() and len(list(rule_jsons_path.glob("*.json"))) > 0:
            self.rule_jsons_dir = rule_jsons_path
        elif rule_infer_path.exists() and len(list(rule_infer_path.glob("*.json"))) > 0:
            self.rule_jsons_dir = rule_infer_path
        elif rule_jsons_path.exists():
            self.rule_jsons_dir = rule_jsons_path
        elif rule_infer_path.exists():
            self.rule_jsons_dir = rule_infer_path
        else:
            self.rule_jsons_dir = rule_jsons_path
        
        # 自动检测模型切割目录（优先选择有文件的目录）
        model_jsons_path = self.dataset_dir / "model_jsons"
        model_infer_path = self.dataset_dir / "model_infer"
        
        if model_jsons_path.exists() and len(list(model_jsons_path.glob("*.json"))) > 0:
            self.model_jsons_dir = model_jsons_path
        elif model_infer_path.exists() and len(list(model_infer_path.glob("*.json"))) > 0:
            self.model_jsons_dir = model_infer_path
        elif model_jsons_path.exists():
            self.model_jsons_dir = model_jsons_path
        elif model_infer_path.exists():
            self.model_jsons_dir = model_infer_path
        else:
            self.model_jsons_dir = model_jsons_path
        
        self.fusion_jsons_dir = self.dataset_dir / "fusion_jsons"
        
        # 输出路径
        self.output_dir = self.dataset_dir
        self.vis_dir = self.dataset_dir / "al_visual"
        
        # 更新全局配置
        global NORMAL_CHAR_RATIO, BORDER_BIAS_THRESH, VALLEY_DEPTH_THRESH
        NORMAL_CHAR_RATIO = self.config.normal_char_ratio
        BORDER_BIAS_THRESH = self.config.border_bias_thresh
        VALLEY_DEPTH_THRESH = self.config.valley_depth_thresh
        
        print(f"[INFO] 主动学习管理器初始化完成")
        print(f"       数据集ID: {dataset_id}")
        print(f"       数据集目录: {self.dataset_dir}")
    
    def _load_json_file(self, file_path: Path) -> Optional[dict]:
        """加载JSON文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] 无法加载文件: {file_path} - {str(e)}")
            return None
    
    def _calculate_scores(self, img_name: str) -> Optional[dict]:
        """
        计算单个图片的主动学习评分
        
        Args:
            img_name: 图片名称（不含扩展名）
        
        Returns:
            评分结果字典，包含 total_score, disagree_score, uncertain_score, abnormal_score
            以及 rule_boxes, model_boxes
        """
        # 构建文件路径
        rule_json_path = self.rule_jsons_dir / f"{img_name}_chars.json"
        model_json_path = self.model_jsons_dir / f"{img_name}_model.json"
        img_path = self.pdf_lines_dir / f"{img_name}.png"
        
        # 检查文件是否存在
        if not rule_json_path.exists():
            print(f"[WARN] 规则切割文件不存在: {rule_json_path}")
            return None
        if not model_json_path.exists():
            print(f"[WARN] 模型切割文件不存在: {model_json_path}")
            return None
        if not img_path.exists():
            print(f"[WARN] 图片文件不存在: {img_path}")
            return None
        
        # 加载切割结果
        rule_data = self._load_json_file(rule_json_path)
        model_data = self._load_json_file(model_json_path)
        
        if not rule_data or not model_data:
            return None
        
        # 提取切割框
        rule_boxes = [(c["col_start"], 0, c["col_end"], 55) for c in rule_data.get("chars", [])]
        model_boxes = [(c["col_start"], 0, c["col_end"], 55) for c in model_data.get("chars", [])]
        
        # 获取模型概率和谷深度
        model_probs = model_data.get("probabilities", [0.5] * len(model_boxes))
        valley_depth = model_data.get("valley_depth", 0.2)
        
        # 计算各项评分
        disagree = count_disagree_score(rule_boxes, model_boxes)
        uncertain = uncertain_score(model_probs)
        abnormal = abnormal_score(rule_boxes, valley_depth)
        
        # 计算总分
        weights = self.config.score_weights
        total_score = disagree * weights["disagree"] + \
                      uncertain * weights["uncertain"] + \
                      abnormal * weights["abnormal"]
        
        return {
            "img_name": img_name,
            "img_path": str(img_path),
            "total_score": round(total_score, 4),
            "disagree_score": disagree,
            "uncertain_score": uncertain,
            "abnormal_score": abnormal,
            "rule_boxes": rule_boxes,
            "model_boxes": model_boxes
        }
    
    def _visualize_compare(self, img_path: str, rule_boxes: list, model_boxes: list, save_path: str):
        """生成规则切割和模型切割的对比图"""
        try:
            _import_pil()
            original_img = Image.open(img_path).convert("RGB")
            img_w, img_h = original_img.size
            purple_color = (128, 0, 128)
            line_height = 1

            rule_result_img = original_img.copy()
            draw_rule = ImageDraw.Draw(rule_result_img)
            for box in rule_boxes:
                draw_rule.rectangle(box, outline="green", width=1)

            model_result_img = original_img.copy()
            draw_model = ImageDraw.Draw(model_result_img)
            for box in model_boxes:
                draw_model.rectangle(box, outline="red", width=1)

            total_canvas_h = img_h * 3 + line_height * 2
            concat_img = Image.new("RGB", (img_w, total_canvas_h), color="white")
            concat_img.paste(original_img, (0, 0))
            line1 = Image.new("RGB", (img_w, line_height), color=purple_color)
            concat_img.paste(line1, (0, img_h))
            concat_img.paste(rule_result_img, (0, img_h + line_height))
            line2 = Image.new("RGB", (img_w, line_height), color=purple_color)
            concat_img.paste(line2, (0, img_h * 2 + line_height))
            concat_img.paste(model_result_img, (0, img_h * 2 + line_height * 2))
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            concat_img.save(save_path)
        except Exception as e:
            print(f"[WARN] 生成可视化图失败: {save_path} - {str(e)}")
    
    def run(self, top_n: int = 100, generate_visual: bool = True) -> List[dict]:
        """
        执行主动学习
        
        Args:
            top_n: 选取评分最高的样本数量，默认100
            generate_visual: 是否生成可视化对比图，默认True
        
        Returns:
            top_samples: 评分最高的样本列表
        """
        print(f"\n[INFO] 开始执行主动学习 (数据集: {self.dataset_id})")
        print(f"       目标样本数: {top_n}")
        
        # 检查目录是否存在
        if not self.dataset_dir.exists():
            print(f"[ERROR] 数据集目录不存在: {self.dataset_dir}")
            return []
        
        if not self.pdf_lines_dir.exists():
            print(f"[ERROR] 行图片目录不存在: {self.pdf_lines_dir}")
            return []
        
        # 获取所有行图片
        img_paths = list(self.pdf_lines_dir.glob("*.png"))
        print(f"[INFO] 发现 {len(img_paths)} 张行图片")
        
        if len(img_paths) == 0:
            print(f"[WARN] 没有找到行图片")
            return []
        
        # 计算每张图片的评分
        all_results = []
        for img_path in img_paths:
            img_name = img_path.stem
            result = self._calculate_scores(img_name)
            if result:
                all_results.append(result)
        
        print(f"[INFO] 成功计算 {len(all_results)} 张图片的评分")
        
        # 按总分排序，取前N个
        all_results.sort(key=lambda x: x["total_score"], reverse=True)
        top_samples = all_results[:top_n]
        
        print(f"[INFO] 筛选完成，选取前 {len(top_samples)} 个样本")
        
        # 保存结果到 top_annotate_samples.json
        output_path = self.output_dir / "top_annotate_samples.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(top_samples, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 结果已保存到: {output_path}")
        
        # 生成可视化对比图
        if generate_visual:
            self.vis_dir.mkdir(parents=True, exist_ok=True)
            for sample in top_samples:
                vis_path = self.vis_dir / f"{sample['img_name']}.png"
                self._visualize_compare(
                    img_path=sample["img_path"],
                    rule_boxes=sample["rule_boxes"],
                    model_boxes=sample["model_boxes"],
                    save_path=vis_path
                )
            print(f"[INFO] 可视化对比图已保存到: {self.vis_dir}")
        
        print(f"\n[SUCCESS] 主动学习执行完成！")
        return top_samples
    
    def copy_top_files(self, top_samples: List[dict], target_dir: Optional[str] = None):
        """
        复制Top样本到指定目录
        
        Args:
            top_samples: 主动学习返回的样本列表
            target_dir: 目标目录，默认在数据集目录下创建 top_samples 子目录
        """
        if target_dir is None:
            target_dir = self.dataset_dir / "top_samples"
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for sample in top_samples:
            img_name = sample["img_name"]
            src_img = Path(sample["img_path"])
            src_rule_json = self.rule_jsons_dir / f"{img_name}_chars.json"
            
            if src_img.exists():
                shutil.copy2(src_img, target_path / src_img.name)
            if src_rule_json.exists():
                shutil.copy2(src_rule_json, target_path / src_rule_json.name)
        
        print(f"[INFO] Top样本已复制到: {target_path}")


# ===================== 便捷函数 =====================
def run_active_learning_for_dataset(dataset_id: str, top_n: int = 100) -> List[dict]:
    """
    便捷函数：为指定数据集执行主动学习
    
    Args:
        dataset_id: 数据集ID
        top_n: 选取的样本数量
    
    Returns:
        top_samples: 评分最高的样本列表
    """
    manager = ActiveLearningManager(dataset_id=dataset_id)
    return manager.run(top_n=top_n)

#python -m bussiness.active_learning_manager --dataset pdf5823 --top-n 100
# ===================== 命令行入口 =====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="主动学习样本选择")
    parser.add_argument("--dataset", required=True, help="数据集ID（如 pdf01、pdf5823）")
    parser.add_argument("--top-n", type=int, default=100, help="选取的样本数量")
    parser.add_argument("--no-visual", action="store_true", help="不生成可视化图")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"主动学习样本选择 (数据集: {args.dataset})")
    print("=" * 60)
    
    manager = ActiveLearningManager(dataset_id=args.dataset)
    top_samples = manager.run(top_n=args.top_n, generate_visual=not args.no_visual)
    
    print("\n" + "=" * 60)
    print(f"完成！共生成 {len(top_samples)} 个待标注样本")
    print("=" * 60)