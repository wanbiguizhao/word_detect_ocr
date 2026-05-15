# Word Detect OCR - 汉字识别与标注系统

## 项目简介

汉字识别与标注系统，用于古籍/文献 PDF 中汉字的自动分割、OCR 识别、聚类标注和主动学习。系统提供 Web 界面，支持人工标注与 AI 辅助标注的闭环工作流。

核心能力：
- PDF 页面行级分割，提取单个汉字图片
- OCR 模型自动预测汉字及置信度
- HOG/ArcFace 特征聚类，将相似汉字归为一组
- 多轮聚类标注，自动排除已标注字符，逐步扩展标注覆盖
- 预标注确认，基于 OCR 预测批量审核
- 统一标注数据模型，跨系统数据一致性保证

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.10+, FastAPI, Uvicorn |
| 前端 | Vue 3, Vite, Ant Design Vue, TypeScript |
| AI - 分割 | U-Net 语义分割模型 |
| AI - 特征 | ArcFace 特征提取 + HOG 特征 |
| AI - 聚类 | HDBSCAN, KMeans, ArcFace 聚类 |
| 数据存储 | JSON 文件（无数据库依赖） |

---

## 项目结构

```
word_detect_ocr/
├── ai/                              # AI 模块
│   ├── segment/                     # 汉字分割（U-Net）
│   │   ├── model.py                 # 分割模型定义
│   │   ├── train.py                 # 训练脚本
│   │   ├── infer.py                 # 推理脚本
│   │   ├── activate_learning.py     # 主动学习
│   │   └── post.py                  # 后处理
│   ├── feature/                     # 特征提取（ArcFace）
│   │   ├── model.py                 # 特征提取模型
│   │   ├── train.py / infer.py      # 训练与推理
│   │   └── loss.py                  # ArcFace Loss
│   └── word_recgonize/              # 汉字识别
│       ├── hog_clusterer.py         # HOG 特征聚类器
│       └── cluster_config.py        # 聚类配置
│
├── server/                          # Web 服务
│   ├── backend/                     # FastAPI 后端
│   │   ├── app.py                   # 应用入口
│   │   ├── start_server.py          # 启动脚本
│   │   ├── config.py                # 配置管理
│   │   ├── models.py                # Pydantic 模型
│   │   ├── utils.py                 # 工具函数
│   │   ├── routers/                 # API 路由
│   │   │   ├── labeling.py          # 标注任务 API
│   │   │   ├── clusters.py          # 聚类 API
│   │   │   ├── annotation.py        # 标注管理 API
│   │   │   ├── multi_clustering.py  # 多轮聚类 API
│   │   │   ├── images.py            # 图片服务 API
│   │   │   ├── line_status.py       # 行状态 API
│   │   │   └── events.py            # SSE 事件推送
│   │   ├── services/                # 业务逻辑
│   │   │   ├── multi_clustering_manager.py  # 多轮聚类管理
│   │   │   └── char_pool_manager.py         # 字符池管理
│   │   ├── datastore/               # 数据存储层
│   │   │   ├── data_store.py        # 统一数据存储（核心同步）
│   │   │   ├── stats_manager.py     # 标注统计
│   │   │   └── sync_logger.py       # 同步日志
│   │   └── scripts/                 # 数据同步脚本
│   │       ├── sync_data.py
│   │       ├── sync_cluster_labels.py
│   │       └── sync_unified_to_prelabels.py
│   │
│   └── frontend/                    # Vue 3 前端
│       └── src/
│           ├── views/               # 页面组件
│           │   ├── HomePage.vue                  # 首页
│           │   ├── LabelingDashboard.vue          # 标注仪表盘
│           │   ├── SegmentLabelPage.vue           # 分割标注页
│           │   ├── OcrLabelPage.vue               # OCR 标注页
│           │   ├── OcrClusterLabelPage.vue        # OCR 聚类标注页
│           │   ├── OcrClusterEditPage.vue         # OCR 聚类编辑页
│           │   ├── PreLabelConfirmPage.vue        # 预标注确认页
│           │   ├── MultiClusteringPage.vue        # 多轮聚类列表页
│           │   ├── MultiClusteringLabelPage.vue   # 多轮聚类标注页
│           │   ├── LineCheckPage.vue              # 行检查页
│           │   └── LineDetailPage.vue             # 行详情页
│           ├── components/          # 公共组件
│           │   └── LineCanvas.vue   # 行画布组件
│           └── router/index.js      # 路由配置
│
├── bussiness/                       # 业务数据
│   ├── datahome/                    # 数据集目录
│   │   └── {dataset_id}/
│   │       ├── pdf_chars/           # 切割后的汉字图片
│   │       ├── clusters/            # 单轮聚类结果
│   │       ├── multi_clustering/    # 多轮聚类数据
│   │       │   ├── char_pool/       # 字符池
│   │       │   │   ├── all_chars.json
│   │       │   │   ├── labeled.json
│   │       │   │   └── unlabeled.json
│   │       │   ├── rounds/          # 各轮次数据
│   │       │   │   └── round_N/
│   │       │   │       ├── hog_clusters.json
│   │       │   │       └── labels.json
│   │       │   └── round_history.json
│   │       ├── pre_labels.json      # OCR 预测结果
│   │       ├── prelabel_status.json # 预标注状态
│   │       └── unified_labels.json  # 统一标注数据
│   └── migration/                   # 数据迁移模块
│
├── docs/                            # 项目文档
│   ├── 预标注确认页面需求规格说明书.md
│   ├── 预标注页面操作数据流图.md
│   ├── 多轮聚类标注页面需求规格说明书.md
│   ├── 多轮聚类标注列表页面需求规格说明书.md
│   ├── 多轮聚类标注列表页面数据流说明书.md
│   ├── 多轮聚类页面操作数据流图.md
│   ├── 多轮聚类数据源扩展方案评估.md
│   ├── 统一标注数据架构设计方案.md
│   └── server架构重构方案.md
│
├── dataset/                         # 训练数据集
├── config.py                        # 全局配置
└── requirements.txt                 # Python 依赖
```

---

## 快速启动

### 1. 安装依赖

```bash
# Python 后端依赖
pip install -r requirements.txt

# 额外依赖（聚类/特征提取）
pip install scikit-learn hdbscan opencv-python numpy

# 前端依赖
cd server/frontend
npm install
```

### 2. 启动后端

```bash
cd server/backend
python start_server.py
# 后端运行在 http://localhost:8000
```

### 3. 启动前端

```bash
cd server/frontend
npm run dev
# 前端运行在 http://localhost:5173
```

### 4. 访问系统

打开浏览器访问 http://localhost:5173

---

## 页面路由

| 路由 | 页面 | 说明 |
|------|------|------|
| `/` | HomePage | 首页导航 |
| `/dashboard` | LabelingDashboard | 标注进度仪表盘 |
| `/segment` | SegmentLabelPage | 分割标注 |
| `/ocr` | OcrLabelPage | OCR 预测浏览 |
| `/ocr-label/:id` | OcrClusterLabelPage | OCR 聚类标注 |
| `/ocr-edit/:id` | OcrClusterEditPage | OCR 聚类编辑 |
| `/prelabel-confirm/:char` | PreLabelConfirmPage | 预标注确认（按汉字） |
| `/multi-clustering` | MultiClusteringPage | 多轮聚类列表 |
| `/mc-label/:round/:cluster_id` | MultiClusteringLabelPage | 多轮聚类标注 |
| `/line-check` | LineCheckPage | 行检查 |
| `/line-detail/:lineName` | LineDetailPage | 行详情 |

---

## 核心工作流

### 标注流程

```
PDF 页面
  │
  ├─→ 行分割（AI segment）→ 单行图片
  │
  ├─→ 字符切割 → 单字图片（pdf_chars/）
  │
  ├─→ OCR 预测（AI feature）→ 预标注（pre_labels.json）
  │
  ├─→ 预标注确认页 → 人工审核/修正 OCR 预测
  │
  ├─→ 特征聚类（HOG/ArcFace）→ 聚类分组
  │
  └─→ 多轮聚类标注 → 逐步标注所有汉字
        │
        ├─ 第1轮：未标注字符聚类 → 标注
        ├─ 第2轮：剩余未标注 / 低置信度 → 标注
        └─ 第N轮：...
```

### 数据一致性

系统维护三个数据视图的同步：

| 系统 | 核心文件 | 说明 |
|------|----------|------|
| OCR 预标注 | `pre_labels.json` + `prelabel_status.json` | OCR 预测结果和状态 |
| 聚类标注 | `rounds/round_N/labels.json` | 各轮次聚类标注 |
| 统一标注 | `unified_labels.json` | 全局唯一标注数据源 |

任何标注操作（确认/修改/跳过/撤回）都会通过 `DataStore` 自动同步到所有相关系统，包括跨轮次更新。

---

## API 概览

| 前缀 | 模块 | 说明 |
|------|------|------|
| `/api/labeling/*` | 标注任务 | 任务管理、优先队列 |
| `/api/clusters/*` | 单轮聚类 | 聚类列表、标注、跳过 |
| `/api/mc/*` | 多轮聚类 | 轮次管理、聚类标注、字符池 |
| `/api/annotation/*` | 标注管理 | 标注查询、统计 |
| `/api/char-images/*` | 图片搜索 | 按汉字搜索图片 |
| `/api/images/*` | 图片服务 | 图片文件访问 |

---

## 配置

后端配置文件：`server/backend/config.json`

```json
{
  "dataset": {
    "current": "pdf5826",
    "source": "pdf01"
  },
  "labeling": {
    "high_confidence_threshold": 0.9,
    "low_confidence_threshold": 0.7
  },
  "clustering": {
    "default_method": "arcface",
    "supported_methods": ["arcface", "hog"]
  }
}
```

---

## 开发日志

- **2026年4月**：重启项目，完成基础架构重构
- **2026年5月**：实现多轮聚类标注系统、预标注确认页面、统一标注数据模型、跨系统数据同步机制
