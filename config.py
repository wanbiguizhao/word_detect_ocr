import os
import sys
from pathlib import Path

def get_project_root(relative_to: str = __file__) -> Path:
    """
    获取项目根目录（绝对路径），不受当前工作目录影响
    
    Args:
        relative_to: 参考文件路径（默认使用当前config.py的路径，确保根目录计算准确）
    
    Returns:
        Path对象：项目根目录的绝对路径
    """
    # 1. 转为Path对象（兼容不同系统路径分隔符）
    ref_path = Path(relative_to).resolve()
    # 2. 向上追溯到项目根目录（可根据实际目录结构调整parents的层级）
    # 若config.py在项目根目录下，parents[0]即为根目录；若在src/下，用parents[1]
    project_root = ref_path.parents[0]
    
    # 3. 验证根目录（可选：检查是否存在核心目录，确保路径正确）
    required_dirs = ["char_image_dataset"]  # 项目核心目录
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            print(f"⚠️  警告：项目根目录未检测到{dir_name}目录，可能根目录计算错误！")
    
    return project_root

# ------------------- 全局根路径（一次定义，全项目复用） -------------------
PROJECT_ROOT = get_project_root()

# ------------------- 基于根路径拼接各子目录/文件路径 -------------------
# 数据集目录
DATASET_ROOT = PROJECT_ROOT / "char_image_dataset"
# 标注文件路径
ANNOTATION_FILE = PROJECT_ROOT / "char_annotation.csv"
# 模型权重保存目录
MODEL_DIR = PROJECT_ROOT / "models"
# 日志/可视化文件目录
VISUAL_DIR = PROJECT_ROOT / "visualizations"

# ------------------- 数据集/训练配置（复用之前的参数） -------------------
# 图片配置
IMG_SIZE = (100, 100)  # 统一尺寸
IMG_CHANNELS = 1       # 灰度图，单通道

# 训练配置
BATCH_SIZE = 32
VAL_SPLIT = 0.2        # 验证集比例
LEARNING_RATE = 1e-4
EPOCHS = 20

# ------------------- 路径初始化函数（自动创建缺失目录） -------------------
def init_project_dirs():
    """初始化项目所需目录（避免运行时因目录不存在报错）"""
    dirs_to_create = [DATASET_ROOT, MODEL_DIR, VISUAL_DIR]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 目录已创建/存在：{dir_path}")

# 初始化目录（导入config.py时自动执行）
init_project_dirs()

# ------------------- 便捷路径转换函数（可选） -------------------
def to_abs_path(relative_path: str) -> str:
    """
    将相对路径转为基于项目根目录的绝对路径（兼容字符串/Path对象）
    
    Args:
        relative_path: 相对路径（如"char_image_dataset/one/one_001.png"）
    
    Returns:
        str：绝对路径字符串（便于传入OpenCV/PyTorch等函数）
    """
    abs_path = PROJECT_ROOT / relative_path
    return str(abs_path.resolve())