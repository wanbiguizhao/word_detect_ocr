import json
import os
from pathlib import Path
from PIL import Image

def extract_chars_from_segmentation(dataset_id):
    """从分割图片中提取字符图片"""
    base_dir = Path(__file__).parent.parent / "bussiness" / "datahome" / dataset_id
    
    clusters_path = base_dir / "clusters" / "hog_clusters.json"
    seg_dir = base_dir / "model_infer"
    chars_dir = base_dir / "pdf_chars"
    lines_dir = base_dir / "pdf_lines"
    
    chars_dir.mkdir(parents=True, exist_ok=True)
    lines_dir.mkdir(parents=True, exist_ok=True)
    
    if not clusters_path.exists():
        print(f"聚类数据不存在: {clusters_path}")
        return
    
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)
    
    extracted_count = 0
    
    for cluster_id, chars in clusters_data.get("clusters", {}).items():
        for char_info in chars:
            char_id = char_info.get("char_id")
            image_path = char_info.get("image_path", "")
            lineage = char_info.get("lineage", {})
            
            # 从 char_id 解析行信息
            # 格式: page_X_line_Y_char_Z
            parts = char_id.split("_")
            if len(parts) >= 4 and parts[0] == "page":
                page_num = parts[1]
                line_idx = parts[3]
                line_name = f"page_{page_num}_line_{line_idx}"
                
                # 查找对应的分割图片
                seg_file = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = seg_dir / f"{line_name}_seg{ext}"
                    if candidate.exists():
                        seg_file = candidate
                        break
                
                if seg_file:
                    try:
                        with Image.open(seg_file) as img:
                            # 尝试从 lineage 获取坐标
                            col_start = lineage.get("col_start", 0)
                            col_end = lineage.get("col_end", img.width)
                            char_width = lineage.get("width", col_end - col_start)
                            
                            # 如果没有坐标信息，使用默认值
                            if col_end <= col_start:
                                col_end = col_start + 40
                            
                            # 计算裁剪区域
                            left = col_start
                            top = 0
                            right = min(col_end, img.width)
                            bottom = img.height
                            
                            if right > left and bottom > top:
                                cropped = img.crop((left, top, right, bottom))
                                char_file = chars_dir / f"{char_id}.png"
                                cropped.save(char_file)
                                extracted_count += 1
                                
                                # 同时保存行图片（如果不存在）
                                line_file = lines_dir / f"{line_name}.png"
                                if not line_file.exists():
                                    img.save(line_file)
                    except Exception as e:
                        print(f"处理 {char_id} 失败: {e}")
    
    print(f"已提取 {extracted_count} 个字符图片")

if __name__ == "__main__":
    extract_chars_from_segmentation("pdf5823")