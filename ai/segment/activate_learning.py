import os
import json
import shutil
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== 【1】基础配置 =====================
NORMAL_CHAR_RATIO = (0.5, 1.8)
BORDER_BIAS_THRESH = 5
VALLEY_DEPTH_THRESH = 0.3
SCORE_WEIGHTS = {
    "disagree": 0.4,
    "uncertain": 0.3,
    "abnormal": 0.3
}

# ===================== 【2】核心工具函数 =====================
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

# ===================== 【3】主动学习评分器 =====================
class ActiveLearningSelector:
    def __init__(self, weights=SCORE_WEIGHTS):
        self.weights = weights

    def calculate_total_score(self, rule_boxes, model_boxes, model_probs, valley_depth):
        disagree = count_disagree_score(rule_boxes, model_boxes)
        uncertain = uncertain_score(model_probs)
        abnormal = abnormal_score(rule_boxes, valley_depth)
        total_score = disagree * self.weights["disagree"] + uncertain * self.weights["uncertain"] + abnormal * self.weights["abnormal"]
        return round(total_score, 4), disagree, uncertain, abnormal

    def select_top_samples(self, all_results, top_n=50):
        sorted_samples = sorted(all_results, key=lambda x: x["total_score"], reverse=True)
        return sorted_samples[:top_n]

# ===================== 【4】可视化函数 =====================
def visualize_compare(img_path, rule_boxes, model_boxes, save_path):
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
    concat_img.save(save_path)
    plt.close()

# ==============================================
# 【✅ 完全独立解耦函数】可在任意地方单独调用
# ==============================================
def copy_top_files(top_samples, rule_result_dir, top_origin_dir):
    """
    独立工具函数：复制Top样本的图片+规则标注到备份目录
    :param top_samples: 主动学习返回的样本列表
    :param rule_result_dir: 规则标注文件目录
    :param top_origin_dir: 自定义备份目录
    """
    top_origin_dir = Path(top_origin_dir)
    top_origin_dir.mkdir(parents=True, exist_ok=True)
    
    for sample in tqdm(top_samples, desc="备份Top样本文件"):
        img_name = sample["img_name"]
        src_img = Path(sample["img_path"])
        src_rule_json = Path(rule_result_dir) / f"{img_name}_chars.json"

        if src_img.exists():
            shutil.copy2(src_img, top_origin_dir / src_img.name)
        if src_rule_json.exists():
            shutil.copy2(src_rule_json, top_origin_dir / src_rule_json.name)

def rename_original_files(top_samples, rule_result_dir):
    """
    独立工具函数：给原文件添加 xdel_ 前缀
    :param top_samples: 主动学习返回的样本列表
    :param rule_result_dir: 规则标注文件目录
    """
    for sample in tqdm(top_samples, desc="原文件添加 xdel_ 前缀"):
        img_name = sample["img_name"]
        src_img = Path(sample["img_path"])
        src_rule_json = Path(rule_result_dir) / f"{img_name}_chars.json"

        if src_img.exists() and not src_img.name.startswith("xdel_"):
            src_img.rename(src_img.parent / f"xdel_{src_img.name}")
        if src_rule_json.exists() and not src_rule_json.name.startswith("xdel_"):
            src_rule_json.rename(src_rule_json.parent / f"xdel_{src_rule_json.name}")

# ===================== 【5】纯主动学习函数（无任何文件操作） =====================
def run_active_learning(
    data_dir,
    model_result_dir,
    rule_result_dir,
    save_dir,
    top_n=500
):
    """
    纯计算/筛选/可视化函数，与文件复制/重命名完全解耦
    :return: top_samples 样本列表（可用于后续文件操作）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / "compare_vis"
    vis_dir.mkdir(exist_ok=True)

    selector = ActiveLearningSelector()
    all_sample_results = []
    img_paths = list(Path(data_dir).glob("*.png"))
    print(f"📊 开始分析 {len(img_paths)} 张图片...")

    for img_path in tqdm(img_paths):
        img_name = img_path.stem
        rule_json = Path(rule_result_dir) / f"{img_name}_chars.json"
        model_json = Path(model_result_dir) / f"{img_name}_model.json"
        if not rule_json.exists() or not model_json.exists():
            continue

        rule_boxes = [ (c["col_start"], 0, c["col_end"], 55) for c in json.load(open(rule_json))["chars"] ]
        model_data = json.load(open(model_json))
        model_boxes = [ (c["col_start"], 0, c["col_end"], 55) for c in model_data["chars"] ]
        model_probs = model_data.get("probabilities", [0.5]*len(model_boxes))
        valley_depth = model_data.get("valley_depth", 0.2)

        total_score, disagree, uncertain, abnormal = selector.calculate_total_score(rule_boxes, model_boxes, model_probs, valley_depth)
        all_sample_results.append({
            "img_name": img_name, "img_path": str(img_path),
            "total_score": total_score, "disagree_score": disagree,
            "uncertain_score": uncertain, "abnormal_score": abnormal,
            "rule_boxes": rule_boxes, "model_boxes": model_boxes
        })

    # 筛选样本
    top_samples = selector.select_top_samples(all_sample_results, top_n)
    print(f"✅ 筛选完成！Top {top_n} 个待标注样本已生成")

    # 保存结果
    with open(save_dir / "top_annotate_samples.json", "w", encoding="utf-8") as f:
        json.dump(top_samples, f, ensure_ascii=False, indent=2)

    # 生成可视化图
    for sample in tqdm(top_samples, desc="生成可视化图"):
        visualize_compare(
            img_path=sample["img_path"], rule_boxes=sample["rule_boxes"],
            model_boxes=sample["model_boxes"], save_path=vis_dir / f"{sample['img_name']}.png"
        )

    print(f"\n🎯 主动学习执行完成！主目录：{save_dir}")
    # ✅ 仅返回样本列表，不执行任何文件操作
    return top_samples

# ===================== 【6】使用示例（分步执行，完全解耦） =====================
if __name__ == "__main__":
    # ========== 1. 基础路径配置 ==========
    DATA_DIR = BASE_DIR / "dataset" / "unlable"
    MODEL_RESULT_DIR = BASE_DIR / "dataset" / "infer_results"
    RULE_RESULT_DIR = BASE_DIR / "dataset" / "unlable"
    CUSTOM_SAVE_DIR = BASE_DIR / "dataset"/"my_active_learning"
    CUSTOM_TOP_ORIGIN_DIR = BASE_DIR /"dataset" /"pre_lable"

    # ========== 2. 第一步：执行主动学习（仅计算、画图、保存结果） ==========
    top_samples = run_active_learning(
        data_dir=DATA_DIR,
        model_result_dir=MODEL_RESULT_DIR,
        rule_result_dir=RULE_RESULT_DIR,
        save_dir=CUSTOM_SAVE_DIR,
        top_n=100
    )

    # ========== 3. 第二步：【单独调用】文件操作（解耦执行） ==========
    # 复制文件到备份目录
    copy_top_files(top_samples, RULE_RESULT_DIR, CUSTOM_TOP_ORIGIN_DIR)
    # 重命名原文件（最后执行）
    #rename_original_files(top_samples, RULE_RESULT_DIR)

    print("\n🎉 全部流程执行完成！")