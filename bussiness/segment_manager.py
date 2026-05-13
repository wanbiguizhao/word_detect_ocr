"""
分割流程管理器
统一编排：PDF -> 图片 -> 行 -> 汉字 的完整流程

功能：
    1. PDF转图片（pdf2image）
    2. 图片转行（image2line）
    3. 行转汉字（line2char）- 支持规则/模型/融合三种模式
    4. 行高度过滤（filter）

使用方法：
    from ai.segment.segment_manager import SegmentManager
    from image_tools.pdf_config import Pdf2ImageConfig
    from image_tools.image_config import Image2LineConfig
    from ai.segment.segment_config import Line2CharConfig
    
    # 创建配置
    pdf_cfg = Pdf2ImageConfig(dpi=300, output_dir="pdf_images")
    img_cfg = Image2LineConfig(output_dir="line_images")
    char_cfg = Line2CharConfig(min_line_height=30, max_line_height=100)
    
    # 创建管理器
    manager = SegmentManager(pdf_cfg, img_cfg, char_cfg)
    
    # 执行完整流程
    lineage = manager.process_pdf("input.pdf")
    
    # 或分步执行
    manager.pdf_to_images("input.pdf")
    manager.image_to_lines("page_1.png")
    manager.filter_lines(line_paths)
    manager.process_lines_by_fusion(valid_lines)
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from image_tools.pdf_config import Pdf2ImageConfig
from image_tools.image_config import Image2LineConfig
from bussiness.segment_config import Line2CharConfig
from ai.segment.line_filter import LineHeightFilter, filter_lines_by_height


@dataclass
class ProcessResult:
    """处理结果封装类"""
    success: bool  # 是否成功
    message: str  # 结果信息
    data: Optional[Dict] = None  # 附加数据


class SegmentManager:
    """
    分割流程管理器
    
    统一管理 PDF -> 图片 -> 行 -> 汉字 的完整处理流程
    
    Attributes:
        pdf_cfg: PDF转图片配置
        img_cfg: 图片转行配置
        char_cfg: 行转汉字配置
    """
    
    def __init__(
        self,
        pdf_cfg: Pdf2ImageConfig,
        img_cfg: Image2LineConfig,
        char_cfg: Line2CharConfig
    ):
        """
        初始化分割流程管理器
        
        Args:
            pdf_cfg: PDF转图片配置对象
            img_cfg: 图片转行配置对象
            char_cfg: 行转汉字配置对象
        """
        self.pdf_cfg = pdf_cfg
        self.img_cfg = img_cfg
        self.char_cfg = char_cfg
        
        # 初始化行高度过滤器
        self._line_filter = LineHeightFilter(
            min_h=char_cfg.min_line_height,
            max_h=char_cfg.max_line_height
        )
    
    def pdf_to_images(self, pdf_path: str) -> Tuple[bool, List[str]]:
        """
        PDF转图片
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            (success, image_paths)
            - success: 是否成功
            - image_paths: 生成的图片路径列表
        """
        try:
            import fitz
            from PIL import Image
            
            config = self.pdf_cfg
            doc = fitz.open(pdf_path)
            image_paths = []
            
            # 处理页码范围
            start_page = 0
            end_page = doc.page_count - 1
            if config.page_range:
                start_page = max(0, config.page_range[0] - 1)
                end_page = min(doc.page_count - 1, config.page_range[1] - 1)
            
            # 逐页转换
            for page_idx in range(start_page, end_page + 1):
                page = doc.load_page(page_idx)
                zoom = config.dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(
                    matrix=mat,
                    alpha=False,
                    colorspace=fitz.csGRAY if config.grayscale else fitz.csRGB
                )
                
                page_num = page_idx + 1
                img_name = f"page_{page_num}.{config.output_format}"
                img_path = os.path.join(str(config.output_dir), img_name)
                os.makedirs(str(config.output_dir), exist_ok=True)
                pix.save(img_path)
                image_paths.append(img_path)
            
            doc.close()
            return True, image_paths
            
        except Exception as e:
            return False, [str(e)]
    
    def image_to_lines(self, img_path: str) -> Tuple[bool, List[Dict]]:
        """
        图片转行

        Args:
            img_path: 图片文件路径

        Returns:
            (success, lines)
            - success: 是否成功
            - lines: 提取的行列表，每项包含路径和位置信息
        """
        try:
            import cv2
            from image_tools.imageCore import (
                CharSegmentConfig, Preprocessor, TextLineDetector
            )

            cfg = CharSegmentConfig()
            pre = Preprocessor(cfg)
            detector = TextLineDetector(cfg)

            proc_img, original, h, w = pre.process(img_path)
            lines = detector.detect(proc_img, h, w)

            if len(lines) == 0:
                return True, []

            os.makedirs(str(self.img_cfg.output_dir), exist_ok=True)

            result_lines = []
            img_name = Path(img_path).stem

            for lid, (y1, y2) in enumerate(lines):
                line_img = original[y1:y2, :]
                line_save = os.path.join(str(self.img_cfg.output_dir), f"{img_name}_line_{lid}.png")
                cv2.imwrite(line_save, line_img)
                result_lines.append({
                    "line_id": lid,
                    "path": line_save,
                    "y_start": int(y1),
                    "y_end": int(y2),
                    "width": int(w),
                    "height": int(y2 - y1)
                })

            return True, result_lines

        except Exception as e:
            return False, [str(e)]
    
    def filter_lines(self, line_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        按行高度过滤
        
        Args:
            line_paths: 行图片路径列表
        
        Returns:
            (valid_lines, invalid_lines)
            - valid_lines: 符合条件的行图片路径
            - invalid_lines: 不符合条件 的行图片路径
        """
        valid, invalid = [], []
        for p in line_paths:
            if self._line_filter.is_valid(p):
                valid.append(p)
            else:
                invalid.append(p)
        return valid, invalid
    
    def process_lines_by_rule(self, line_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        规则执行：行转汉字（纯规则方法）
        
        Args:
            line_paths: 行图片路径列表
        
        Returns:
            处理结果字典 {行路径: 结果列表}
        """
        import cv2
        import json
        from image_tools.imageCore import CharSegmentConfig, VerticalProjectionSegmenter
        
        cfg = CharSegmentConfig()
        seg = VerticalProjectionSegmenter(cfg)
        
        # 确保输出目录存在
        os.makedirs(str(self.char_cfg.rule_json_dir), exist_ok=True)
        
        results = {}
        for line_path in line_paths:
            try:
                line_name = Path(line_path).stem
                rule_json = str(Path(self.char_cfg.rule_json_dir) / f"{line_name}_chars.json")
                
                # 读取行图片
                gray = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    print(f"   [WARN] 无法读取图片: {line_path}")
                    continue
                
                h, w = gray.shape[:2]
                if w == 0 or h == 0:
                    print(f"   [WARN] 空图片: {line_path}")
                    continue
                
                # 获取分段结果
                segments = seg.get_segment_classes(gray)
                
                # 转换为 build_ordered_list 期望的格式
                chars = []  # 汉字列表，每个元素包含 col_start, col_end, width
                segments_type_start_end = []  # 所有分段 [(type, start, end), ...]
                
                for seg_type, start, end in segments:
                    segments_type_start_end.append([int(seg_type), int(start), int(end)])
                    if seg_type == 1:
                        chars.append({
                            "col_start": int(start),
                            "col_end": int(end),
                            "width": int(end - start)
                        })
                
                # 调试：显示处理信息
                if len(chars) == 0:
                    print(f"   [DEBUG] 未找到汉字: {line_name} (segments: {len(segments)})")
                
                # 保存规则切割JSON（符合 build_ordered_list 期望的格式）
                rule_result = {
                    "image_path": line_path,
                    "line_name": line_name,
                    "chars": chars,
                    "segments_type_start_end": segments_type_start_end,
                    "total_chars": len(chars),
                    "image_width": w,
                    "image_height": h,
                    "total_segments": len(segments)
                }
                
                # 调试：打印保存信息
                print(f"   [DEBUG] 保存规则JSON: {rule_json} (chars: {len(chars)})")
                
                # 强制写入并刷新到磁盘
                with open(rule_json, 'w', encoding='utf-8') as f:
                    json.dump(rule_result, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                
                # 验证文件是否写入成功
                if os.path.exists(rule_json) and os.path.getsize(rule_json) > 0:
                    print(f"   [DEBUG] 文件写入成功: {os.path.getsize(rule_json)} bytes")
                    results[line_path] = rule_result
                else:
                    print(f"   [ERROR] 文件写入失败: {rule_json}")
                
            except Exception as e:
                print(f"   [ERROR] 处理失败: {line_name} - {str(e)}")
                continue
        
        return results
    
    def process_lines_by_model(self, line_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        模型执行：行转汉字（纯模型方法）
        
        Args:
            line_paths: 行图片路径列表
        
        Returns:
            处理结果字典 {行路径: [boxes, json_path]}
        """
        from ai.segment.infer_rebuild import CharSegmentInfer
        
        infer = CharSegmentInfer()
        results = {}
        
        for line_path in line_paths:
            try:
                save_dir = self.char_cfg.model_json_dir
                boxes, json_path = infer.infer_single_image(
                    line_path, save_dir, save_vis=self.char_cfg.save_visual
                )
                results[line_path] = [{"boxes": boxes, "json": json_path}]
            except Exception as e:
                print(f"   [ERROR] 模型推理失败: {Path(line_path).stem} - {str(e)}")
                continue
        
        return results
    
    def process_lines_by_fusion(self, line_paths: List[str]) -> Dict[str, Dict]:
        """
        融合执行：行转汉字（规则+模型融合）

        Args:
            line_paths: 行图片路径列表

        Returns:
            处理结果字典 {行路径: fusion_data}
        """
        from ai.segment.post import process_single_image, FusionConfig

        # 创建融合配置
        cfg = FusionConfig()
        cfg.RULE_JSON_DIR = Path(self.char_cfg.rule_json_dir)
        cfg.MODEL_JSON_DIR = Path(self.char_cfg.model_json_dir)
        cfg.OUTPUT_JSON_DIR = Path(self.char_cfg.output_json_dir)
        cfg.OUTPUT_IMG_DIR = Path(self.char_cfg.output_json_dir).parent / "visual"
        
        # 确保输出目录存在
        os.makedirs(str(cfg.OUTPUT_JSON_DIR), exist_ok=True)
        os.makedirs(str(cfg.OUTPUT_IMG_DIR), exist_ok=True)

        results = {}
        for line_path in line_paths:
            try:
                line_name = Path(line_path).stem
                rule_json = str(Path(self.char_cfg.rule_json_dir) / f"{line_name}_chars.json")
                model_json = str(Path(self.char_cfg.model_json_dir) / f"{line_name}_model.json")
                
                if os.path.exists(rule_json) and os.path.exists(model_json):
                    output_json = str(Path(self.char_cfg.output_json_dir) / f"{line_name}_fusion.json")
                    # 使用 FusionConfig 方式调用
                    process_single_image(Path(line_path), cfg)
                    
                    with open(output_json, 'r', encoding='utf-8') as f:
                        fusion_data = json.load(f)
                    results[line_path] = fusion_data
                else:
                    print(f"   [DEBUG] 缺少文件: {line_name} (rule: {os.path.exists(rule_json)}, model: {os.path.exists(model_json)})")
            except Exception as e:
                print(f"   [ERROR] 融合失败: {line_name} - {str(e)}")
                continue
        
        return results
    
    def extract_char_images(self, line_path: str, fusion_data: Dict) -> List[Dict]:
        """
        从行图片中提取单个汉字图片

        Args:
            line_path: 行图片路径
            fusion_data: 融合结果数据

        Returns:
            汉字图片信息列表，每项包含路径和血缘信息
        """
        import cv2
        
        char_results = []
        line_name = Path(line_path).stem
        
        # 解析行名获取页面和行信息
        page_num = None
        line_idx = None
        if '_line_' in line_name:
            parts = line_name.split('_')
            for i, part in enumerate(parts):
                if part == 'page' and i + 1 < len(parts):
                    page_num = int(parts[i + 1])
                elif part == 'line' and i + 1 < len(parts):
                    line_idx = int(parts[i + 1])
        
        # 读取行图片
        line_img = cv2.imread(line_path)
        if line_img is None:
            print(f"   [ERROR] 无法读取行图片: {line_path}")
            return char_results
        
        # 确保输出目录存在
        char_output_dir = Path(self.char_cfg.output_dir)
        os.makedirs(str(char_output_dir), exist_ok=True)
        
        # 提取汉字
        fusion_chars = fusion_data.get('fusion_chars', [])
        for char_idx, char_info in enumerate(fusion_chars):
            if char_info.get('type') != 'CHAR':
                continue
            
            # 获取字符边界
            start = char_info.get('start', 0)
            end = char_info.get('end', 0)
            
            # 切割汉字图片
            char_img = line_img[:, start:end]
            if char_img.size == 0:
                continue
            
            # 生成汉字图片路径
            char_name = f"{line_name}_char_{char_idx}.png"
            char_path = str(char_output_dir / char_name)
            
            # 保存汉字图片
            cv2.imwrite(char_path, char_img)
            
            # 记录血缘信息
            char_result = {
                'char_id': f"{line_name}_char_{char_idx}",
                'path': char_path,
                'line_path': line_path,
                'line_name': line_name,
                'page_num': page_num,
                'line_idx': line_idx,
                'char_idx': char_idx,
                'col_start': start,
                'col_end': end,
                'width': char_info.get('width', 0),
                'prob': char_info.get('prob', 0.0),
                'is_merged': char_info.get('is_merged', False),
                'is_garbage': char_info.get('is_garbage', False)
            }
            char_results.append(char_result)
        
        return char_results
    
    def process_pdf(self, pdf_path: str, skip_filter: bool = False) -> Dict:
        """
        完整流程：PDF -> 图片 -> 行 -> 汉字
        
        Args:
            pdf_path: PDF文件路径
            skip_filter: 是否跳过行高度过滤，默认False
        
        Returns:
            血缘索引字典
        """
        lineage = {
            "metadata": {
                "version": "1.0",
                "source_pdf": pdf_path,
                "total_pages": 0,
                "total_lines": 0,
                "total_chars": 0,
                "created_at": str(pdf_path)
            },
            "pages": {},
            "lines": {},
            "chars": {}
        }
        
        # 步骤1: PDF转图片
        print(f"[1/5] PDF to images: {pdf_path}")
        success, image_paths = self.pdf_to_images(pdf_path)
        if not success:
            print(f"Error: {image_paths}")
            return lineage
        
        print(f"   Generated {len(image_paths)} images")
        
        # 记录页面血缘
        for img_path in image_paths:
            img_name = Path(img_path).stem
            page_num = int(img_name.replace('page_', '')) if img_name.startswith('page_') else None
            if page_num:
                lineage["pages"][page_num] = {
                    "page_num": page_num,
                    "image_path": img_path,
                    "lines": []
                }
        
        # 步骤2: 图片转行
        print(f"[2/5] Image to lines...")
        all_line_paths = []
        for img_path in image_paths:
            success, lines = self.image_to_lines(img_path)
            if success:
                img_name = Path(img_path).stem
                page_num = int(img_name.replace('page_', '')) if img_name.startswith('page_') else None
                for line in lines:
                    if "path" in line:
                        all_line_paths.append(line["path"])
                        # 记录行血缘
                        line_name = Path(line["path"]).stem
                        lineage["lines"][line_name] = {
                            "line_name": line_name,
                            "path": line["path"],
                            "page_num": page_num,
                            "y_start": line.get("y_start"),
                            "y_end": line.get("y_end"),
                            "width": line.get("width"),
                            "height": line.get("height"),
                            "chars": []
                        }
                        # 关联到页面
                        if page_num and page_num in lineage["pages"]:
                            lineage["pages"][page_num]["lines"].append(line_name)
        
        print(f"   Extracted {len(all_line_paths)} lines")
        
        # 步骤3: 行高度过滤
        if skip_filter:
            valid_lines = all_line_paths
            invalid_lines = []
        else:
            valid_lines, invalid_lines = self.filter_lines(all_line_paths)
        
        print(f"   Filtered: {len(valid_lines)} valid, {len(invalid_lines)} rejected")
        
        # 步骤4: 规则切割
        print(f"[3/5] Rule-based segmentation...")
        rule_results = self.process_lines_by_rule(valid_lines)
        print(f"   Generated {len(rule_results)} rule JSONs (total lines: {len(valid_lines)})")
        
        # 步骤5: 模型推理
        print(f"[4/5] Model inference...")
        # 限制处理行数，避免运行时间过长
        test_limit = 0
        lines_to_process = valid_lines[:test_limit] if test_limit > 0 else valid_lines
        print(f"   Testing with {len(lines_to_process)} lines (total: {len(valid_lines)})")
        self.process_lines_by_model(lines_to_process)
        print(f"   Generated model JSONs")
        
        # 步骤6: 行转汉字（融合）
        print(f"[5/5] Fusion (rule + model)...")
        # 只融合已生成模型JSON的行
        lines_to_fusion = valid_lines[:test_limit] if test_limit > 0 else valid_lines
        print(f"   Fusion with {len(lines_to_fusion)} lines")
        fusion_results = self.process_lines_by_fusion(lines_to_fusion)
        
        # 步骤7: 汉字图片切割
        print(f"[6/6] Extracting character images...")
        all_char_results = []
        for line_path, fusion_data in fusion_results.items():
            char_results = self.extract_char_images(line_path, fusion_data)
            all_char_results.extend(char_results)
            
            # 更新行血缘中的汉字列表
            line_name = Path(line_path).stem
            if line_name in lineage["lines"]:
                lineage["lines"][line_name]["chars"] = [c["char_id"] for c in char_results]
        
        # 记录汉字血缘
        for char_result in all_char_results:
            char_id = char_result["char_id"]
            lineage["chars"][char_id] = char_result
        
        total_chars = len(all_char_results)
        print(f"   Extracted {total_chars} character images")
        
        # 更新血缘索引
        lineage["metadata"]["total_pages"] = len(image_paths)
        lineage["metadata"]["total_lines"] = len(valid_lines)
        lineage["metadata"]["total_chars"] = total_chars
        
        # 保存血缘索引到文件
        lineage_path = str(Path(self.char_cfg.output_dir).parent / "lineage.json")
        with open(lineage_path, 'w', encoding='utf-8') as f:
            json.dump(lineage, f, ensure_ascii=False, indent=2)
        print(f"   Lineage saved to: {lineage_path}")
        
        return lineage
    
    def summarize(self, lineage: Dict) -> None:
        """
        输出血缘索引统计信息
        
        Args:
            lineage: 血缘索引字典
        """
        meta = lineage.get("metadata", {})
        print(f"\n=== Process Summary ===")
        print(f"Pages: {meta.get('total_pages', 0)}")
        print(f"Lines: {meta.get('total_lines', 0)}")
        print(f"Chars: {meta.get('total_chars', 0)}")


class ClusterManager:
    """
    汉字图片聚类管理器
    
    负责对segment_manager生成的汉字图片进行HOG聚类
    """
    
    def __init__(self, input_dir: str, output_dir: str, n_clusters: int = 50, lineage_file: str = None):
        """
        初始化聚类管理器
        
        Args:
            input_dir: 汉字图片输入目录（必需）
            output_dir: 聚类结果输出目录（必需）
            n_clusters: 聚类数量，默认50
            lineage_file: 血缘关系文件路径（可选，用于关联聚类结果）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.n_clusters = n_clusters
        self.lineage_file = Path(lineage_file) if lineage_file else None
        
        self.clusters = {}
        self.char_to_cluster = {}
        self.lineage_data = {}
        
        self._load_lineage()
    
    def _load_lineage(self) -> None:
        """
        加载血缘关系数据
        
        血缘关系文件包含汉字与原始PDF页面、行的对应关系，用于聚类结果的溯源分析。
        若未提供血缘文件路径，则跳过加载。
        """
        import json
        
        if not self.lineage_file:
            print(f"[INFO] 未指定血缘文件路径，跳过血缘关系加载")
            self.lineage_data = {}
            return
        
        print(f"[INFO] 尝试加载血缘关系文件: {self.lineage_file}")
        
        if self.lineage_file.exists():
            try:
                with open(self.lineage_file, 'r', encoding='utf-8') as f:
                    self.lineage_data = json.load(f)
                
                chars_count = len(self.lineage_data.get('chars', {}))
                pages_count = len(self.lineage_data.get('pages', {}))
                lines_count = len(self.lineage_data.get('lines', {}))
                
                print(f"[INFO] 血缘关系加载成功:")
                print(f"       - 文件路径: {self.lineage_file}")
                print(f"       - 汉字数量: {chars_count}")
                print(f"       - 页面数量: {pages_count}")
                print(f"       - 行数量: {lines_count}")
                
            except Exception as e:
                print(f"[ERROR] 加载血缘文件失败: {self.lineage_file}")
                print(f"       错误信息: {str(e)}")
                self.lineage_data = {}
        else:
            print(f"[WARN] 血缘文件不存在: {self.lineage_file}")
            print(f"       将跳过血缘关系关联")
            self.lineage_data = {}
    
    def _get_char_lineage(self, char_id: str) -> Dict:
        """
        获取单个汉字的血缘信息
        
        Args:
            char_id: 汉字ID
        
        Returns:
            血缘信息字典
        """
        return self.lineage_data.get("chars", {}).get(char_id, {})
    
    def cluster(self) -> Dict:
        """
        执行HOG聚类
        
        Returns:
            聚类结果字典
        """
        import cv2
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        print("=" * 60)
        print("HOG汉字图片聚类")
        print("=" * 60)
        
        char_files = list(self.input_dir.glob("*_char_*.png"))
        print(f"[INFO] 发现 {len(char_files)} 张汉字图片")
        
        if len(char_files) == 0:
            print("[WARN] 没有可用的汉字图片")
            return {}
        
        print(f"[INFO] 正在提取HOG特征...")
        features = []
        char_info_list = []
        
        target_size = (64, 64)
        
        for img_file in char_files:
            try:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                img = cv2.resize(img, target_size)
                
                hog = cv2.HOGDescriptor(
                    _winSize=target_size,
                    _blockSize=(16, 16),
                    _blockStride=(8, 8),
                    _cellSize=(8, 8),
                    _nbins=9
                )
                
                hog_features = hog.compute(img).flatten()
                features.append(hog_features)
                char_info_list.append({
                    "char_id": img_file.stem,
                    "image_path": str(img_file)
                })
            except Exception as e:
                print(f"[WARN] 处理图片失败 {img_file.name}: {str(e)}")
        
        if len(features) == 0:
            print("[ERROR] 无法提取任何图片特征")
            return {}
        
        print(f"[INFO] 正在标准化特征...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        print(f"[INFO] 正在执行K-means聚类 (n_clusters={self.n_clusters})...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features_scaled)
        
        self.clusters = {}
        self.char_to_cluster = {}
        
        for idx, char_info in enumerate(char_info_list):
            cluster_id = int(labels[idx])
            char_id = char_info["char_id"]
            
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            
            self.clusters[cluster_id].append(char_info)
            self.char_to_cluster[char_id] = cluster_id
        
        print(f"[INFO] 聚类完成，共生成 {len(self.clusters)} 个聚类")
        
        cluster_sizes = sorted([(k, len(v)) for k, v in self.clusters.items()], 
                             key=lambda x: x[1], reverse=True)
        print(f"[INFO] 聚类大小分布（前10个）:")
        for cluster_id, size in cluster_sizes[:10]:
            print(f"       聚类 {cluster_id}: {size} 个汉字")
        
        return {
            "clusters": self.clusters,
            "char_to_cluster": self.char_to_cluster
        }
    
    def save_results(self) -> None:
        """
        保存聚类结果到文件
        
        标注目录结构：
        clusters/
        ├── hog_clusters.json          # 聚类结果（包含血缘信息）
        ├── cluster_images/            # 聚类示例图片
        │   ├── cluster_0/
        │   ├── cluster_1/
        │   └── ...
        └── labeling/                  # 标注目录
            ├── labels.json             # 标注结果：cluster_id -> 汉字
            ├── unlabeled/              # 未标注的聚类
            │   ├── cluster_0/
            │   └── ...
            └── labeled/                # 已标注的聚类
                └── ...
        """
        import cv2
        import json
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        clusters_with_lineage = {}
        for cluster_id, chars in self.clusters.items():
            clusters_with_lineage[cluster_id] = []
            for char_info in chars:
                char_id = char_info["char_id"]
                lineage_info = self._get_char_lineage(char_id)
                clusters_with_lineage[cluster_id].append({
                    "char_id": char_id,
                    "image_path": char_info["image_path"],
                    "lineage": lineage_info
                })
        
        result = {
            "config": {
                "n_clusters": self.n_clusters,
                "total_chars": sum(len(chars) for chars in self.clusters.values()),
                "total_clusters": len(self.clusters)
            },
            "clusters": clusters_with_lineage,
            "char_to_cluster": self.char_to_cluster
        }
        
        json_path = self.output_dir / "hog_clusters.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 聚类结果已保存到: {json_path}")
        
        cluster_images_dir = self.output_dir / "cluster_images"
        cluster_images_dir.mkdir(exist_ok=True)
        
        labeling_dir = self.output_dir / "labeling"
        unlabeled_dir = labeling_dir / "unlabeled"
        labeled_dir = labeling_dir / "labeled"
        
        for cluster_id, chars in self.clusters.items():
            cluster_dir = cluster_images_dir / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            
            for idx, char_info in enumerate(chars[:5]):
                src_path = Path(char_info["image_path"])
                dst_path = cluster_dir / f"{idx}_{src_path.name}"
                
                try:
                    img = cv2.imread(str(src_path))
                    if img is not None:
                        cv2.imwrite(str(dst_path), img)
                except:
                    pass
            
            unlabeled_cluster_dir = unlabeled_dir / f"cluster_{cluster_id}"
            unlabeled_cluster_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, char_info in enumerate(chars):
                src_path = Path(char_info["image_path"])
                dst_path = unlabeled_cluster_dir / f"{idx}_{src_path.name}"
                
                try:
                    img = cv2.imread(str(src_path))
                    if img is not None:
                        cv2.imwrite(str(dst_path), img)
                except:
                    pass
        
        labels_json_path = labeling_dir / "labels.json"
        labels_data = {}
        for cluster_id in self.clusters.keys():
            labels_data[str(cluster_id)] = {
                "char": None,
                "status": "unlabeled",
                "confidence": None
            }
        
        with open(labels_json_path, 'w', encoding='utf-8') as f:
            json.dump(labels_data, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 标注目录已创建: {labeling_dir}")
        print(f"[INFO] 标注结果文件: {labels_json_path}")
        print(f"[INFO] 聚类示例图片已保存到: {cluster_images_dir}")
    
    def run(self) -> Dict:
        """
        执行完整的聚类流程
        
        Returns:
            聚类结果字典
        """
        result = self.cluster()
        self.save_results()
        
        print("=" * 60)
        print("聚类完成！")
        print("=" * 60)
        
        return result

def run_segment(data_base_path, pdf_path):
    """
    运行PDF文本分割流程
    
    Args:
        data_base_path (Path): 数据基础目录，用于存放输出文件
        pdf_path (Path): PDF文件路径
    
    Returns:
        None
        
    输出目录结构（相对于 data_base_path）:
        pdf_images/       - PDF转换后的页面图片
        pdf_lines/        - 切割后的行图片
        pdf_chars/        - 切割后的汉字图片
        rule_infer/       - 规则切割结果JSON
        model_infer/      - 模型切割结果JSON
        fusion/           - 融合结果
            fusion_json/  - 融合结果JSON
            fusion_visual/ - 融合可视化图片
        lineage.json      - 血缘关系文件
    
    注意：每个PDF文件夹应有独立的 data_base_path，避免不同PDF处理结果互相覆盖。
    """
    # 获取项目根目录（用于定位模型文件）
    project_root = Path(__file__).resolve().parent.parent
    
    # PDF转图片配置
    pdf_cfg = Pdf2ImageConfig()
    pdf_cfg.output_dir = data_base_path / "pdf_images"
    
    # 图片转行配置
    img_cfg = Image2LineConfig()
    img_cfg.output_dir = data_base_path / "pdf_lines"
    
    # 行转汉字配置
    char_cfg = Line2CharConfig()
    char_cfg.max_line_height = 50 
    char_cfg.min_line_height = 40 
    char_cfg.output_dir = data_base_path / "pdf_chars"
    char_cfg.rule_json_dir = data_base_path / "rule_infer"
    char_cfg.model_json_dir = data_base_path / "model_infer"
    char_cfg.output_json_dir = data_base_path / "fusion" / "fusion_json"
    char_cfg.output_img_dir = data_base_path / "fusion" / "fusion_visual"
    
    # 设置模型路径（从项目根目录定位）
    char_cfg.model_ae_path = str(project_root / "ai/model_storage/feature_model.pth")
    char_cfg.model_segment_path = str(project_root / "ai/model_storage/char_segment_classifier_0427.pth")
    
    # 验证配置
    char_cfg.validate()
    
    # 创建所有输出目录
    os.makedirs(pdf_cfg.output_dir, exist_ok=True)
    os.makedirs(img_cfg.output_dir, exist_ok=True)
    os.makedirs(char_cfg.output_dir, exist_ok=True)
    os.makedirs(char_cfg.rule_json_dir, exist_ok=True)
    os.makedirs(char_cfg.model_json_dir, exist_ok=True)
    os.makedirs(char_cfg.output_json_dir, exist_ok=True)
    os.makedirs(char_cfg.output_img_dir, exist_ok=True)
    
    print(f"[INFO] 数据基础目录: {data_base_path}")
    print(f"[INFO] PDF文件: {pdf_path}")
    print(f"[INFO] 输出目录已准备就绪")
    
    # 执行分割
    segment_manager = SegmentManager(pdf_cfg=pdf_cfg, img_cfg=img_cfg, char_cfg=char_cfg)
    segment_manager.process_pdf(str(pdf_path))    
if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent
    data_base_path = base_dir / "bussiness"/"datahome"/ "pdf5823"
    pdf_path = data_base_path / "gwyb195823.pdf"
    run_segment(data_base_path, pdf_path)
    # print("\n" + "=" * 60)
    # print("开始聚类")
    # print("=" * 60)
    
    # cluster_manager = ClusterManager(
    #     input_dir=data_base_path / "pdf_chars",
    #     output_dir=data_base_path / "clusters",
    #     n_clusters=1000
    # )
    # cluster_manager.run()
    