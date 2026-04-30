"""
汉字图片筛选模块
基于融合模型结果，从行图片中提取并筛选高质量汉字图片

使用方法：
    from ai.segment.filter import CharFilter
    
    # 创建筛选器
    filter = CharFilter()
    
    # 批量筛选
    report = filter.batch_filter("fusion_dir", "line_img_dir")
    
    # 自定义配置
    filter.config.min_char_width = 20
    filter.config.min_prob_threshold = 0.7
"""

import json
import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np

# 设置基础路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

class FilterConfig:
    """汉字图片筛选配置"""
    def __init__(self):
        # 宽度筛选（汉字通常在20-80像素之间）
        self.min_char_width = 10      # 最小宽度阈值（放宽到10px，兼容小字符）
        self.max_char_width = 100     # 最大宽度阈值（放宽到100px，兼容合并字符）
        
        # 模型概率筛选
        self.min_prob_threshold = 0.5 # 最小概率阈值（放宽到0.5）
        self.skip_prob_check = False  # 是否跳过概率检查
        
        # 合并状态筛选
        self.allow_merged = True      # 是否允许合并后的字符
        self.allow_single = True      # 是否允许非合并字符
        
        # 图像质量筛选
        self.min_content_ratio = 0.05 # 最小内容比例（非空白像素占比）
        self.max_content_ratio = 0.95 # 最大内容比例
        self.skip_content_check = True # 是否跳过内容比例检查（默认跳过）
        
        # 输出配置
        self.OUTPUT_GOOD_DIR = Path("filtered_chars/good")
        self.OUTPUT_BAD_DIR = Path("filtered_chars/bad")
        self.OUTPUT_JSON_PATH = Path("filtered_chars/filter_report.json")
        
        # 创建目录
        for dir_path in [self.OUTPUT_GOOD_DIR, self.OUTPUT_BAD_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

class CharFilter:
    """汉字图片筛选器"""
    
    def __init__(self, config: FilterConfig = None):
        """初始化筛选器"""
        self.config = config if config else FilterConfig()
    
    def load_fusion_result(self, json_path: str) -> Dict:
        """加载融合结果JSON"""
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def extract_char_images(
        self,
        line_img_path: str,
        chars: List[Dict],
        save_dir: Path,
        prefix: str = ""
    ) -> List[Tuple[str, Dict]]:
        """从行图片中提取单个汉字图片"""
        results = []
        try:
            img = Image.open(line_img_path).convert("L")  # 转为灰度图
            img_height = img.height
            
            for char in chars:
                x_start = char["col_start"]
                x_end = char["col_end"]
                y_start = char.get("abs_y_start", 0)
                y_end = char.get("abs_y_end", img_height)
                
                # 裁剪单个汉字
                char_img = img.crop((x_start, y_start, x_end + 1, y_end))
                
                # 生成保存路径
                char_id = char.get("char_id", len(results))
                save_path = save_dir / f"{prefix}_char_{char_id}.png"
                char_img.save(save_path)
                
                results.append((str(save_path), char))
        except Exception as e:
            print(f"[ERROR] 提取字符图片失败: {e}")
        
        return results
    
    def calculate_content_ratio(self, img_path: str) -> float:
        """计算图像内容比例（非空白像素占比）"""
        try:
            img = Image.open(img_path).convert("L")
            img_array = np.array(img)
            # 空白定义：接近白色的像素（>240）
            white_pixels = np.sum(img_array > 240)
            total_pixels = img_array.size
            content_ratio = 1 - (white_pixels / total_pixels)
            return content_ratio
        except Exception as e:
            print(f"[WARN] 计算内容比例失败: {e}")
            return 0.0
    
    def filter_char(
        self,
        char_info: Dict,
        model_probs: List[float] = None
    ) -> Tuple[bool, str]:
        """
        筛选单个字符
        返回：(是否通过筛选, 筛选失败原因)
        """
        cfg = self.config
        
        # 1. 宽度筛选
        width = char_info.get("width", 0)
        if width < cfg.min_char_width:
            return False, f"宽度过小 ({width}px < {cfg.min_char_width}px)"
        if width > cfg.max_char_width:
            return False, f"宽度过大 ({width}px > {cfg.max_char_width}px)"
        
        # 2. 合并状态筛选
        is_merged = char_info.get("is_merged", False)
        if is_merged and not cfg.allow_merged:
            return False, "不允许合并字符"
        if not is_merged and not cfg.allow_single:
            return False, "不允许非合并字符"
        
        # 3. 模型概率筛选（可跳过）
        if not cfg.skip_prob_check:
            prob = char_info.get("prob", 0.0)
            if prob > 0 and prob < cfg.min_prob_threshold:
                return False, f"概率过低 ({prob:.4f} < {cfg.min_prob_threshold})"
        
        # 4. 乱码标记筛选
        is_garbage = char_info.get("is_garbage", False)
        if is_garbage:
            return False, "标记为乱码"
        
        # 5. 图像质量筛选（可跳过）
        img_path = char_info.get("save_path")
        if img_path and not cfg.skip_content_check:
            content_ratio = self.calculate_content_ratio(img_path)
            if content_ratio < cfg.min_content_ratio:
                return False, f"内容比例过低 ({content_ratio:.4f} < {cfg.min_content_ratio})"
            if content_ratio > cfg.max_content_ratio:
                return False, f"内容比例过高 ({content_ratio:.4f} > {cfg.max_content_ratio})"
        
        return True, "通过"
    
    def filter_chars_from_fusion(
        self,
        fusion_json_path: str,
        line_img_path: str,
        model_probs: List[float] = None
    ) -> Dict:
        """
        从融合结果中筛选汉字图片
        返回：筛选报告
        """
        cfg = self.config
        
        # 加载融合结果
        fusion_result = self.load_fusion_result(fusion_json_path)
        
        # 从 chars 字段获取字符基础信息（裁剪用）
        chars = fusion_result.get("chars", [])
        
        # 从 fusion_chars 字段获取详细筛选信息
        fusion_chars = fusion_result.get("fusion_chars", [])
        fusion_chars = [c for c in fusion_chars if c.get("type") == "CHAR"]
        
        # 构建筛选信息映射：通过位置匹配
        filter_info_map = {}
        for fc in fusion_chars:
            key = (fc["start"], fc["end"])
            filter_info_map[key] = {
                "prob": fc.get("prob", 0.0),
                "is_garbage": fc.get("is_garbage", False),
                "is_merged": fc.get("is_merged", False)
            }
        
        # 合并字符信息：从chars获取基础信息，从fusion_chars获取筛选信息
        merged_chars = []
        for idx, char in enumerate(chars):
            start = char["col_start"]
            end = char["col_end"]
            key = (start, end)
            
            merged_char = {
                "char_id": idx,
                "col_start": start,
                "col_end": end,
                "width": char["width"],
                **filter_info_map.get(key, {})
            }
            merged_chars.append(merged_char)
        
        # 提取并筛选字符
        good_chars = []
        bad_chars = []
        filter_stats = {
            "total_chars": len(merged_chars),
            "passed": 0,
            "failed": 0,
            "fail_reasons": {}
        }
        
        # 提取字符图片
        prefix = Path(fusion_json_path).stem.replace("_fusion", "")
        extracted = self.extract_char_images(line_img_path, merged_chars, cfg.OUTPUT_BAD_DIR, prefix)
        
        # 逐一筛选
        for img_path, char_info in extracted:
            char_info["save_path"] = img_path
            passed, reason = self.filter_char(char_info, model_probs)
            
            if passed:
                # 移动到good目录
                good_path = cfg.OUTPUT_GOOD_DIR / Path(img_path).name
                # 如果目标文件已存在，先删除
                if good_path.exists():
                    os.remove(good_path)
                os.rename(img_path, good_path)
                good_chars.append({
                    "char_id": char_info.get("char_id"),
                    "width": char_info.get("width"),
                    "prob": char_info.get("prob"),
                    "is_merged": char_info.get("is_merged"),
                    "save_path": str(good_path)
                })
                filter_stats["passed"] += 1
            else:
                bad_chars.append({
                    "char_id": char_info.get("char_id"),
                    "width": char_info.get("width"),
                    "prob": char_info.get("prob"),
                    "is_merged": char_info.get("is_merged"),
                    "reason": reason,
                    "save_path": img_path
                })
                filter_stats["failed"] += 1
                filter_stats["fail_reasons"][reason] = filter_stats["fail_reasons"].get(reason, 0) + 1
        
        # 生成筛选报告
        report = {
            "source": {
                "fusion_json": fusion_json_path,
                "line_image": line_img_path
            },
            "stats": filter_stats,
            "good_chars": good_chars,
            "bad_chars": bad_chars,
            "filter_config": {
                "min_char_width": cfg.min_char_width,
                "max_char_width": cfg.max_char_width,
                "min_prob_threshold": cfg.min_prob_threshold,
                "allow_merged": cfg.allow_merged,
                "allow_single": cfg.allow_single,
                "min_content_ratio": cfg.min_content_ratio,
                "max_content_ratio": cfg.max_content_ratio
            }
        }
        
        return report
    
    def batch_filter(self, fusion_dir: str, line_img_dir: str) -> Dict:
        """
        批量筛选多个融合结果
        
        参数：
            fusion_dir: 融合结果JSON目录路径
            line_img_dir: 行图片目录路径
        
        返回：汇总筛选报告
        """
        cfg = self.config
        fusion_dir = Path(fusion_dir)
        line_img_dir = Path(line_img_dir)
        
        # 获取所有融合结果JSON
        fusion_files = sorted(fusion_dir.glob("*_fusion.json"))
        
        if not fusion_files:
            print("[ERROR] 未找到融合结果JSON文件")
            return {}
        
        print(f"[INFO] 开始批量筛选：共 {len(fusion_files)} 个融合结果")
        
        all_reports = []
        total_passed = 0
        total_failed = 0
        
        for fusion_file in fusion_files:
            print(f"\n处理 {fusion_file.name}...")
            
            # 获取对应的行图片
            stem = fusion_file.stem.replace("_fusion", "")
            line_img_path = line_img_dir / f"{stem}.png"
            
            if not line_img_path.exists():
                print(f"[WARN] 行图片不存在：{line_img_path}")
                continue
            
            # 加载模型概率（如果存在）
            model_json_path = fusion_dir.parent / "infer_results" / f"{stem}_model.json"
            model_probs = []
            if model_json_path.exists():
                with open(model_json_path, "r", encoding="utf-8") as f:
                    model_data = json.load(f)
                    model_probs = model_data.get("probabilities", [])
            
            # 执行筛选
            report = self.filter_chars_from_fusion(str(fusion_file), str(line_img_path), model_probs)
            
            all_reports.append(report)
            total_passed += report["stats"]["passed"]
            total_failed += report["stats"]["failed"]
            
            print(f"  Passed: {report['stats']['passed']} | Failed: {report['stats']['failed']}")
        
        # 保存汇总报告
        summary_report = {
            "total_files": len(all_reports),
            "total_chars_processed": sum(r["stats"]["total_chars"] for r in all_reports),
            "total_passed": total_passed,
            "total_failed": total_failed,
            "fail_reasons_summary": {},
            "reports": all_reports
        }
        
        # 汇总失败原因
        for report in all_reports:
            for reason, count in report["stats"]["fail_reasons"].items():
                summary_report["fail_reasons_summary"][reason] = \
                    summary_report["fail_reasons_summary"].get(reason, 0) + count
        
        # 保存报告
        cfg.OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] 筛选完成！")
        print(f"   总计处理: {total_passed + total_failed} 个字符")
        print(f"   通过筛选: {total_passed} 个")
        print(f"   未通过筛选: {total_failed} 个")
        print(f"   报告已保存: {cfg.OUTPUT_JSON_PATH}")
        
        return summary_report

# ====================== 入口函数 ======================
if __name__ == "__main__":
    from ai.segment.config import BASE_DIR
    
    # 创建筛选器
    char_filter = CharFilter()
    
    # 配置筛选参数（可根据实际需求调整）
    char_filter.config.min_char_width = 10
    char_filter.config.max_char_width = 100
    char_filter.config.min_prob_threshold = 0.5
    char_filter.config.skip_content_check = True
    
    # 设置路径
    FUSION_DIR = BASE_DIR / "dataset" / "fusion_lable"
    LINE_IMG_DIR = BASE_DIR / "dataset" / "pre_lable"
    
    # 执行批量筛选
    char_filter.batch_filter(FUSION_DIR, LINE_IMG_DIR)
