import json
import os
import math
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
from PIL import Image, ImageDraw

# ====================== 1. 配置类（复用原有逻辑，补充批量配置） ======================
class FusionConfig:
    """融合规则配置 + 可视化配置 + 批量配置"""
    def __init__(self):
        # 原有核心配置（完全保留你的超参数，未做任何修改）
        self.narrow_blank_threshold = 15      # 仅空白宽度≤15像素时才合并
        self.min_char_width = 30             # 最小有效汉字宽度（触发合并的阈值）
        self.model_prob_threshold = 0.65     # 模型概率阈值（判断是否为汉字）
        
        # 可视化样式
        self.sep_line_color = (128, 0, 128)  # 紫色分隔线
        self.sep_line_width = 2              # 分隔线 2像素
        self.cut_line_width = 1              # 切割线 1像素
        self.start_color = (255, 0, 0)       # 开始位置：红色
        self.end_color = (0, 255, 0)         # 结束位置：绿色

        # 批量处理目录配置（可根据实际需求修改）
        self.IMAGE_DIR = Path("raw_images")          # 原始图片目录（你可修改）
        self.RULE_JSON_DIR = Path("rule_jsons")      # 规则JSON目录（命名：文件名_chars.json）
        self.MODEL_JSON_DIR = Path("model_jsons")    # 模型JSON目录（命名：文件名_model.json）
        self.OUTPUT_JSON_DIR = Path("fusion_results/json")  # 合并结果JSON保存目录
        self.OUTPUT_IMG_DIR = Path("fusion_results/visual")  # 可视化图片保存目录

# ====================== 2. 工具函数（复用+补充批量适配） ======================
def load_json(file_path: str) -> Dict:
    """加载JSON文件（增加更友好的异常提示）"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"加载JSON失败 {file_path}：{str(e)}")

def save_json(data: Dict, save_path: str):
    """保存JSON文件"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def mean_prob(probs: List[float]) -> float:
    """计算指定区间内模型概率的均值（积分）"""
    return float(np.mean(probs)) if probs and len(probs) > 0 else 0.0

def get_all_image_files(img_dir: Path) -> List[Path]:
    """获取目录下所有图片文件（支持常见格式）"""
    img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    img_files = []
    for ext in img_extensions:
        img_files.extend(img_dir.glob(f"*{ext}"))
        img_files.extend(img_dir.glob(f"*{ext.upper()}"))
    return sorted(img_files)

# ====================== 3. 原有核心逻辑（仅新增移除乱码函数，其余完全保留） ======================
def build_ordered_list(chars: List[Dict], segments: List[List]) -> List[Dict]:
    """构建 汉字+空白 有序列表，并校验相邻元素位置的连续性"""
    elements = []
    # 添加汉字
    for c in chars:
        elements.append({
            "type": "CHAR", "start": c["col_start"], "end": c["col_end"],
            "width": c["width"], "raw": c, "is_merged": False,
            "merged_from": [], "attached_blanks": []
        })
    # 添加空白（仅type=0的空白段）
    for seg in segments:
        t, s, e = seg
        if t == 0:
            elements.append({
                "type": "BLANK", "start": s, "end": e,
                "width": e-s+1, "raw": seg, "is_attached": False
            })
    
    # 按start位置排序
    sorted_elements = sorted(elements, key=lambda x: x["start"])
    
    # 连续性校验（抛出异常，主流程可捕获）
    for i in range(1, len(sorted_elements)):
        prev_elem = sorted_elements[i-1]
        curr_elem = sorted_elements[i]
        
        expected_curr_start = prev_elem["end"] + 1
        actual_curr_start = curr_elem["start"]
        
        if actual_curr_start != expected_curr_start:
            print(
                f"⚠️ 元素位置不连续！\n"
                f"前一个元素（类型：{prev_elem['type']}，raw：{prev_elem['raw']}）：end={prev_elem['end']}\n"
                f"当前元素（类型：{curr_elem['type']}，raw：{curr_elem['raw']}）：start={actual_curr_start}（预期：{expected_curr_start}）"
            )
    
    return sorted_elements

def merge_chars(elements: List[Dict], cfg: FusionConfig, model_probs: List[float]) -> List[Dict]:
    """核心逻辑：窄空白合并汉字（基于空白段概率 + 字符宽度校验）"""
    res, i = [], 0
    total = len(elements)
    max_prob_idx = len(model_probs) - 1 if model_probs else 0
    
    while i < total:
        curr = elements[i]
        if curr["type"] != "BLANK":
            res.append(curr)
            i += 1
            continue
        
        blank = curr
        if blank["width"] > cfg.narrow_blank_threshold:
            res.append(blank)
            i += 1
            continue
        
        has_left = i > 0 and res[-1]["type"] == "CHAR"
        has_right = (i+1) < total and elements[i+1]["type"] == "CHAR"
        if not (has_left and has_right):
            res.append(blank)
            i += 1
            continue
        
        # ===== 新增：字符宽度校验逻辑 =====
        left_char = res[-1]
        right_char = elements[i+1]
        # 仅当左边或右边字符宽度小于min_char_width时，才继续合并逻辑
        left_char_too_narrow = left_char["width"] < cfg.min_char_width
        right_char_too_narrow = right_char["width"] < cfg.min_char_width
        if not (left_char_too_narrow or right_char_too_narrow):
            res.append(blank)
            i += 1
            continue
        
        # 计算空白段概率
        blank_start = blank["start"]
        blank_end = blank["end"]
        prob_start = max(0, blank_start)
        prob_end = min(max_prob_idx, blank_end)
        blank_prob_segment = model_probs[prob_start : prob_end + 1] if model_probs else []
        blank_prob_mean = mean_prob(blank_prob_segment)
        
        if blank_prob_mean < cfg.model_prob_threshold:
            res.append(blank)
            i += 1
            continue
        
        # 执行合并
        left_char = res.pop()
        right_char = elements[i+1]
        merged = {
            "type": "CHAR", "start": left_char["start"], "end": right_char["end"],
            "width": right_char["end"] - left_char["start"] + 1,
            "raw": None,
            "is_merged": True, "merged_from": [left_char["raw"], right_char["raw"]],
            "merged_blank": blank,
            "blank_prob_mean": round(blank_prob_mean, 4),
            "attached_blanks": [],
            # 新增调试字段：记录触发合并的宽度信息
            "merge_trigger": {
                "left_char_width": left_char["width"],
                "right_char_width": right_char["width"],
                "min_char_width": cfg.min_char_width,
                "left_triggered": left_char_too_narrow,
                "right_triggered": right_char_too_narrow
            }
        }
        res.append(merged)
        i += 2
    
    return res

def attach_blanks(elements: List[Dict]) -> List[Dict]:
    """空白挂载到上一个汉字"""
    res, last_char = [], None
    for elem in elements:
        if elem["type"] == "CHAR":
            res.append(elem)
            last_char = elem
        elif elem["type"] == "BLANK" and last_char:
            elem["is_attached"] = True
            last_char["attached_blanks"].append(elem)
    return res

def check_garbage(elements: List[Dict], probs: List[float], cfg: FusionConfig):
    """乱码校验（基于模型概率）"""
    for elem in elements:
        if elem["type"] != "CHAR":
            continue
        start = max(0, elem["start"])
        end = min(len(probs)-1, elem["end"]) if probs else elem["end"]
        p = probs[start : end+1] if probs else []
        elem["prob"] = round(mean_prob(p),4)
        elem["is_garbage"] = elem["prob"] < cfg.model_prob_threshold

# ===== 新增：移除乱码字符的核心函数（仅新增，不修改原有逻辑） =====
def remove_garbage_chars(elements: List[Dict]) -> List[Dict]:
    """
    移除标记为乱码的字符（is_garbage=True）
    同时清理关联的空白段（避免空白段残留导致位置不连续）
    """
    # 1. 过滤掉乱码字符
    clean_elements = []
    removed_count = 0
    for elem in elements:
        if elem["type"] == "CHAR" and elem.get("is_garbage", False):
            removed_count += 1
            continue  # 跳过乱码字符，不加入新列表
        clean_elements.append(elem)
    
    # 2. 重新挂载空白段（因为移除字符后，空白段的挂载关系可能失效）
    clean_elements = attach_blanks(clean_elements)
    
    print(f"[INFO] 移除了 {removed_count} 个乱码字符")
    return clean_elements

def draw_visual_on_original(
    original_img_path: str,
    rule_boxes: List[Tuple[int,int]],
    model_boxes: List[Tuple[int,int]],
    fusion_boxes: List[Tuple[int,int]],
    save_path: str,
    cfg: FusionConfig,
    model_probs: List[Tuple[int, float]] = []
):
    """四层对比可视化"""
    if not Path(original_img_path).exists():
        print(f"[ERROR] 原始图片不存在：{original_img_path}")
        return
    
    try:
        base_img = Image.open(original_img_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] 加载图片失败：{e}")
        return
    
    W, H = base_img.size
    sep = cfg.sep_line_width
    
    total_height = H * 4 + sep * 3
    canvas = Image.new("RGB", (W, total_height), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    current_y = 0

    # 1. 原始图片
    canvas.paste(base_img, (0, current_y))
    current_y += H

    # 紫色分隔线 1
    draw.line([(0, current_y), (W, current_y)], fill=cfg.sep_line_color, width=sep)
    current_y += sep

    # 2. 原图 + 规则切割结果
    rule_img = base_img.copy()
    d_rule = ImageDraw.Draw(rule_img)
    for s, e in rule_boxes:
        d_rule.line([(s, 0), (s, H)], fill=cfg.start_color, width=cfg.cut_line_width)
        d_rule.line([(e, 0), (e, H)], fill=cfg.end_color, width=cfg.cut_line_width)
    canvas.paste(rule_img, (0, current_y))
    current_y += H

    # 紫色分隔线 2
    draw.line([(0, current_y), (W, current_y)], fill=cfg.sep_line_color, width=sep)
    current_y += sep

    # 3. 原图 + 模型切割结果
    model_img = base_img.copy()
    d_model = ImageDraw.Draw(model_img)
    for s, e in model_boxes:
        d_model.line([(s, 0), (s, H)], fill=cfg.start_color, width=cfg.cut_line_width)
        d_model.line([(e, 0), (e, H)], fill=cfg.end_color, width=cfg.cut_line_width)
    canvas.paste(model_img, (0, current_y))
    current_y += H

    # 紫色分隔线 3
    draw.line([(0, current_y), (W, current_y)], fill=cfg.sep_line_color, width=sep)
    current_y += sep

    # 4. 原图 + 融合切割结果（已移除乱码）
    fusion_img = base_img.copy()
    d_fusion = ImageDraw.Draw(fusion_img)
    for s, e in fusion_boxes:
        d_fusion.line([(s, 0), (s, H)], fill=cfg.start_color, width=cfg.cut_line_width)
        d_fusion.line([(e, 0), (e, H)], fill=cfg.end_color, width=cfg.cut_line_width)
    canvas.paste(fusion_img, (0, current_y))

    # 5. 绘制概率可视化（新增）
    prob_height = 50
    prob_canvas = Image.new("RGB", (W, prob_height), (255, 255, 255))
    prob_draw = ImageDraw.Draw(prob_canvas)
    
    # 先构建「列位置→概率值」的映射（处理可能的列缺失）
    prob_dict = {col: prob for col, prob in model_probs}
    for col in range(W):  # 遍历每一列
        prob = prob_dict.get(col, 0.0)  # 无预测的列概率记为0
        prob_percent = prob * 100  # 转百分比（0~100）

        # 计算黄色条高度：96-100%→50像素，91-95%→49像素，以此类推
        # 公式：高度 = 50 - ((100 - 概率值) // 5)，最小为0
        bar_height = math.ceil(prob_percent*0.5)
        if bar_height > 0:
            # 黄色条绘制范围：y从H到H+bar_height-1（因为是闭区间）
            y_start = prob_height-bar_height
            y_end = prob_height
            # 绘制黄色竖线（每列的概率条）
            prob_draw.line([(col, y_start), (col, y_end)], fill=(255, 255, 0), width=1)
    
    # 红色分隔线
    prob_draw.line([(0, 0), (W, 0)], fill=(255, 0, 0), width=1)
    
    # 将概率图拼接到主图下方
    new_canvas = Image.new("RGB", (W, total_height + prob_height), (255, 255, 255))
    new_canvas.paste(canvas, (0, 0))
    new_canvas.paste(prob_canvas, (0, total_height))

    # 保存图片
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    new_canvas.save(save_path)
    print(f"[INFO] 可视化图已保存：{save_path}")

# ====================== 4. 单文件处理函数（仅修改乱码处理部分，其余保留） ======================
def process_single_image(img_path: Path, cfg: FusionConfig) -> bool:
    """
    处理单张图片的完整流程：
    1. 读取规则JSON（文件名_chars.json）
    2. 读取模型JSON（文件名_model.json）
    3. 执行融合逻辑
    4. 保存合并结果+可视化图片
    返回：是否处理成功（bool）
    """
    img_stem = img_path.stem  # 图片文件名（不含扩展名）
    img_name = img_path.name  # 图片完整名（含扩展名）
    
    # 1. 构建各文件路径
    rule_json_path = cfg.RULE_JSON_DIR / f"{img_stem}_chars.json"
    model_json_path = cfg.MODEL_JSON_DIR / f"{img_stem}_model.json"
    output_json_path = cfg.OUTPUT_JSON_DIR / f"{img_stem}_fusion.json"
    output_img_path = cfg.OUTPUT_IMG_DIR / f"{img_stem}_visual.png"

    # 2. 校验必要文件
    if not rule_json_path.exists():
        print(f"[ERROR] 跳过 {img_name}：规则JSON不存在 {rule_json_path}")
        return False
    
    # 3. 加载规则数据
    try:
        rule_data = load_json(rule_json_path)
    except Exception as e:
        print(f"[ERROR] 跳过 {img_name}：加载规则JSON失败 {e}")
        return False
    
    # 4. 加载模型数据（兼容不存在的情况）
    model_data = {}
    model_probs = []
    model_chars = []
    if model_json_path.exists():
        try:
            model_data = load_json(model_json_path)
            model_probs = model_data.get("probabilities", [])
            model_chars = model_data.get("chars", [])
            cfg.model_prob_threshold = model_data.get("threshold", 0.65)
        except Exception as e:
            print(f"⚠️ {img_name}：加载模型JSON失败，使用默认值 {e}")
    else:
        print(f"⚠️ {img_name}：模型JSON不存在 {model_json_path}，使用规则边框替代")

    # 5. 执行融合逻辑
    try:
        ordered = build_ordered_list(rule_data["chars"], rule_data["segments_type_start_end"])
    except Exception as e:
        print(f"[ERROR] 跳过 {img_name}：构建有序列表失败 {e}")
        return False
    
    merged = merge_chars(ordered, cfg, model_probs)
    final = attach_blanks(merged)
    
    # 乱码校验 + 新增：移除乱码字符
    if model_probs:
        check_garbage(final, model_probs, cfg)
        # ===== 仅新增这一行：调用移除乱码函数 =====
        final = remove_garbage_chars(final)

    # 6. 提取各类边框（可视化用 - 仅包含非乱码字符）
    rule_boxes = [(c["col_start"], c["col_end"]) for c in rule_data["chars"]]
    model_boxes = [(c["col_start"], c["col_end"]) for c in model_chars] if model_chars else rule_boxes
    # ===== 修改：过滤掉乱码字符，避免可视化绘制 =====
    fusion_boxes = [(c["start"], c["end"]) for c in final if c["type"] == "CHAR" and not c.get("is_garbage", False)]

    # 7. 生成可视化图片（新增概率可视化）
    prob_tuples = [(i, prob) for i, prob in enumerate(model_probs)]
    draw_visual_on_original(str(img_path), rule_boxes, model_boxes, fusion_boxes, str(output_img_path), cfg, prob_tuples)

    # 8. 构建最终结果（含新增chars字段 - 仅保留非乱码字符）
    new_chars_list = []
    # ===== 修改：仅保留非乱码的CHAR类型 =====
    for char in final:
        if char["type"] == "CHAR" and not char.get("is_garbage", False):
            new_chars_list.append({
                "col_start": char["start"],
                "col_end": char["end"],
                "width": char["width"]
            })
    
    result = {
        "line_info": {
            "line_id": rule_data.get("line_id", 0),
            "parent_image": rule_data.get("parent_image", img_name),
            "line_y_start": rule_data.get("line_y_start", 0),
            "line_y_end": rule_data.get("line_y_end", 0)
        },
        "count_info": {
            "原始字符数": rule_data.get("total_chars", 0),
            # ===== 新增统计：移除乱码后的最终字符数 =====
            "最终字符数": len(new_chars_list)
        },
        "fusion_chars": final,
        "chars": new_chars_list  # 仅保留非乱码字符
    }

    # 9. 保存融合结果JSON
    save_json(result, output_json_path)
    
    # 10. 同时生成 dataset/v2/ 格式数据
    try:
        from dataset.v2_manager import V2DataManager
        v2_manager = V2DataManager()
        
        # 获取行图片路径（从规则数据中获取或从配置目录中查找）
        line_img_path = rule_data.get("line_image_path")
        if not line_img_path or not os.path.exists(line_img_path):
            # 尝试从默认目录查找
            line_img_path = str(cfg.IMAGE_DIR / img_name)
        
        if os.path.exists(line_img_path):
            v2_manager.process_fusion_result(result, line_img_path)
            print(f"   已同步至 dataset/v2/")
    except Exception as e:
        print(f"   [WARN] 同步至 dataset/v2/ 失败：{str(e)}")
    
    print(f"[INFO] 完成 {img_name}：结果已保存至 {output_json_path}")
    return True

# ====================== 5. 批量处理主函数（完全保留） ======================
def batch_process(cfg: FusionConfig):
    """批量处理所有图片"""
    # 1. 获取所有图片文件
    img_files = get_all_image_files(cfg.IMAGE_DIR)
    if not img_files:
        print(f"⚠️ 未在 {cfg.IMAGE_DIR} 找到任何图片文件！")
        return
    
    # 2. 统计变量
    total = len(img_files)
    success = 0
    failed = 0
    failed_list = []

    # 3. 批量执行
    print(f"\n🚀 开始批量处理：共 {total} 张图片")
    print("-" * 80)
    
    for idx, img_path in enumerate(img_files, 1):
        print(f"\n[{idx}/{total}] 处理 {img_path.name}...")
        try:
            if process_single_image(img_path, cfg):
                success += 1
            else:
                failed += 1
                failed_list.append(img_path.name)
        except Exception as e:
            print(f"[ERROR] 处理 {img_path.name} 异常：{str(e)}")
            failed += 1
            failed_list.append(img_path.name)
    
    # 4. 输出统计结果
    print("\n" + "-" * 80)
    print(f"📊 批量处理完成：")
    print(f"   总计：{total} 张")
    print(f"   成功：{success} 张")
    print(f"   失败：{failed} 张")
    if failed_list:
        print(f"   失败列表：{', '.join(failed_list)}")

# ====================== 6. 入口函数（完全保留） ======================
if __name__ == "__main__":
    # 初始化配置
    config = FusionConfig()
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

    config.IMAGE_DIR = PROJECT_DIR / "dataset" / "pre_lable"
    config.MODEL_JSON_DIR = PROJECT_DIR / "dataset" / "infer_results"
    config.RULE_JSON_DIR = PROJECT_DIR / "dataset" / "pre_lable"
    config.OUTPUT_JSON_DIR = PROJECT_DIR / "dataset" / "fusion_lable"
    config.OUTPUT_IMG_DIR = PROJECT_DIR / "dataset" / "fusion_lable"

    # 打印目录配置（方便核对）
    print("📁 目录配置：")
    print(f"   原始图片：{config.IMAGE_DIR.absolute()}")
    print(f"   规则JSON：{config.RULE_JSON_DIR.absolute()}")
    print(f"   模型JSON：{config.MODEL_JSON_DIR.absolute()}")
    print(f"   结果JSON：{config.OUTPUT_JSON_DIR.absolute()}")
    print(f"   可视化图：{config.OUTPUT_IMG_DIR.absolute()}")
    
    # 执行批量处理
    batch_process(config)