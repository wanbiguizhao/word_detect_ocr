# myOcr/__init__.py
import sys
from pathlib import Path

# 1. 初始化项目根路径（包内所有模块可通过 myOcr.PROJECT_ROOT 调用）
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# 2. 导出核心模块/配置（简化导入）
from . import config
from .config import DATASET_ROOT, ANNOTATION_FILE, BATCH_SIZE

# 3. 导出子包（可选，让外部可直接 import myOcr.ai）
from . import ai