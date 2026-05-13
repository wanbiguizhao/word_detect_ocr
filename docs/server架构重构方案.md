# Server 功能架构文档

## 1. 整体架构

```
server/backend/
├── app.py                    # 主应用入口，FastAPI初始化，路由注册
├── config.py                 # 配置管理
├── utils.py                  # 工具函数
├── models.py                 # Pydantic数据模型
├── routers/                  # API路由层
│   ├── images.py            # 图片相关API
│   ├── labeling.py          # 标注相关API
│   ├── multi_clustering.py  # 多轮聚类API
│   ├── clusters.py          # 聚类相关API（旧）
│   ├── annotation.py        # 标注相关API（旧）
│   ├── events.py            # SSE事件API
│   ├── line_status.py       # 行状态API
│   └── pseudo_labels.py     # 伪标签API
├── services/                # 业务逻辑层
│   ├── multi_clustering_manager.py  # 多轮聚类管理
│   └── char_pool_manager.py         # 字符池管理
├── datastore/               # 数据存储层
│   ├── data_store.py        # 统一数据存储抽象层
│   ├── stats_manager.py     # 统计管理
│   └── sync_logger.py       # 同步日志
└── frontend/               # 前端静态文件
```

## 2. 模块职责

### 2.1 routers/ - API路由层
负责处理HTTP请求/响应，参数验证，调用service层

### 2.2 services/ - 业务逻辑层
核心业务逻辑实现：
- **multi_clustering_manager**: 多轮聚类算法、聚类结果管理
- **char_pool_manager**: 字符池管理、已标注数据排除

### 2.3 datastore/ - 数据存储层
数据访问抽象：
- **data_store**: 统一数据存储，支持多数据源同步
- **stats_manager**: 标注统计管理
- **sync_logger**: 同步操作日志

## 3. API接口详细列表

### 3.1 图片相关 API (images.py)

#### GET /api/images/{image_id}/raw
获取原始图片
- **参数**: image_id (路径参数)
- **响应**: 图片二进制数据

#### GET /api/images/{image_id}/detail
获取图片详情
- **参数**: image_id (路径参数)
- **响应**: 
```json
{
  "id": "xxx",
  "image_name": "xxx.png",
  "image_url": "/api/images/{id}/raw",
  "is_annotated": false,
  "is_postponed": false,
  "rule_lines": [...],
  "model_lines": [...],
  "fusion_lines": [...],
  "annotation": {"lines": [...]}
}
```

#### POST /api/images/{image_id}/annotate
保存标注
- **参数**: image_id (路径参数), AnnotationSubmit (body)
- **Body**:
```json
{
  "lines": [
    {"pos": 0, "color": "red"},
    {"pos": 100, "color": "blue"}
  ]
}
```
- **响应**: `{"code": 0, "msg": "保存成功"}`

#### POST /api/images/{image_id}/postpone
标记图片为暂不标注
- **参数**: image_id (路径参数)
- **响应**: `{"code": 0, "msg": "已标记为暂不标注"}`

#### GET /api/char-images/{image_name}
获取字符图片
- **参数**: image_name (路径参数)
- **响应**: 图片二进制数据

#### GET /api/image/pdf_chars/{char_id}
获取PDF字符图片
- **参数**: char_id (路径参数)
- **响应**: 图片二进制数据

---

### 3.2 标注相关 API (labeling.py)

#### GET /api/labeling/stats
获取标注统计
- **响应**:
```json
{
  "dataset": "pdf5823",
  "total_images": 5823,
  "labeled_count": 3000,
  "prelabeled_count": 2000,
  "unlabeled_count": 823,
  "char_stats": [
    {"char": "的", "total": 500, "labeled": 300, "prelabeled": 200}
  ]
}
```

#### GET /api/labeling/char-list
获取字符列表（分页）
- **参数**: 
  - page: int (默认1)
  - page_size: int (默认20)
  - search: str (可选，搜索字符)
  - dataset: str (默认pdf5823)
  - sort_by: str (可选)
  - sort_order: str (默认desc)
- **响应**:
```json
{
  "code": 0,
  "data": [...],
  "total": 100,
  "page": 1,
  "page_size": 20
}
```

#### GET /api/labeling/prelabels/{char}
获取字符的预标注列表
- **参数**:
  - char: str (路径参数)
  - page: int (默认1)
  - page_size: int (默认20)
  - confidence_min: float (可选)
  - dataset: str (默认pdf5823)
- **响应**:
```json
{
  "code": 0,
  "char": "的",
  "total": 500,
  "confirmed": 300,
  "pending": 200,
  "data": [...]
}
```

#### POST /api/labeling/confirm
确认单个标注
- **Body**:
```json
{
  "char_id": "xxx",
  "char": "的"
}
```
- **响应**: `{"code": 0, "msg": "成功确认图片 xxx 为 的"}`

#### POST /api/labeling/modify
修改标注字符
- **Body**:
```json
{
  "char_id": "xxx",
  "char": "了"
}
```

#### POST /api/labeling/confirm/batch
批量确认标注
- **Body**:
```json
{
  "items": [
    {"char_id": "xxx", "char": "的"},
    {"char_id": "yyy", "char": "是"}
  ]
}
```
- **响应**: `{"code": 0, "msg": "成功确认 2/2 个标注", "success_count": 2}`

#### POST /api/labeling/prelabels/confirm
确认预标注
- **Body**:
```json
{
  "char": "的",
  "image_paths": ["path1", "path2"]
}
```

#### POST /api/labeling/prelabels/modify
修改预标注
- **Body**:
```json
{
  "image_path": "xxx",
  "new_char": "了"
}
```

#### POST /api/labeling/prelabels/skip
跳过预标注
- **Body**: `["path1", "path2"]` (List[str])

#### GET /api/labeling/clusters
获取聚类列表
- **参数**: method: str (默认"simple_char")

#### GET /api/labeling/clusters/{cluster_id}
获取聚类详情
- **参数**: cluster_id, method

#### POST /api/labeling/clusters/{cluster_id}/label
标注聚类
- **参数**: cluster_id, char, image_indices (可选), method

#### POST /api/labeling/clusters/run
运行聚类
- **参数**: min_samples: int (默认3)

#### GET /api/labeling/sync/logs
获取同步日志
- **参数**: limit (默认100), date (可选)

#### GET /api/labeling/sync/logs/today
获取今日同步日志

#### GET /api/labeling/sync/stats
获取同步统计

---

### 3.3 多轮聚类 API (multi_clustering.py)

#### GET /api/mc/rounds
获取所有聚类轮次
- **响应**:
```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "rounds": [
      {
        "round": 1,
        "date": "2026-05-01T10:00:00",
        "description": "第1轮聚类",
        "n_clusters": 50,
        "total_chars": 1000,
        "labeled_chars": 300
      }
    ]
  }
}
```

#### POST /api/mc/rounds
启动新一轮聚类
- **Body**:
```json
{
  "method": "hdbscan",
  "n_clusters": null,
  "description": "第2轮聚类",
  "min_cluster_size": 5,
  "min_samples": 2,
  "max_cluster_size": 100
}
```
- **响应**: `{"code": 0, "msg": "success", "round": 2, "message": "第2轮聚类已启动（方法: hdbscan）"}`

#### GET /api/mc/rounds/{round_num}
获取指定轮次信息
- **响应**:
```json
{
  "code": 0,
  "msg": "success",
  "round": 2,
  "clusters": {...},
  "total_clusters": 50,
  "total_chars": 1000,
  "progress": {...}
}
```

#### GET /api/mc/rounds/{round_num}/clusters
获取指定轮次的聚类列表
- **响应**:
```json
{
  "code": 0,
  "msg": "success",
  "round": 2,
  "clusters": [
    {
      "cluster_id": "0",
      "char_count": 25,
      "labeled_count": 10,
      "status": "unlabeled",
      "char": null,
      "char_counts": {"的": 5, "是": 3},
      "confidence": null
    }
  ],
  "total": 50
}
```

#### GET /api/mc/rounds/{round_num}/clusters/{cluster_id}
获取聚类详情
- **响应**:
```json
{
  "code": 0,
  "msg": "success",
  "round": 2,
  "cluster_id": "0",
  "status": "unlabeled",
  "char": null,
  "confidence": null,
  "chars": [
    {
      "index": 0,
      "char_id": "page_1_line_2_char_5",
      "line_name": "page_1_line_2",
      "col_start": 100,
      "col_end": 120,
      "label": null,
      "labeled_at": null
    }
  ],
  "total": 25
}
```

#### POST /api/mc/rounds/{round_num}/clusters/{cluster_id}/labels
保存单个标注
- **Body**:
```json
{
  "charIndex": 0,
  "char": "的"
}
```

#### POST /api/mc/rounds/{round_num}/clusters/{cluster_id}/labels/batch
批量保存标注
- **Body**:
```json
{
  "labels": [
    {"charIndex": 0, "char": "的"},
    {"charIndex": 1, "char": "的"}
  ]
}
```

#### POST /api/mc/rounds/{round_num}/clusters/{cluster_id}/skip
跳过聚类

#### GET /api/mc/char-pool
获取字符池统计
- **响应**:
```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "total": 5000,
    "labeled": 3000,
    "unlabeled": 2000
  }
}
```

#### POST /api/mc/char-pool/init
初始化字符池

#### GET /api/mc/char-pool/unlabeled
获取未标注字符列表（分页）
- **参数**: page, page_size

#### GET /api/mc/rounds/{round_num}/progress
获取轮次进度

#### GET /api/mc/unified-labels/count
获取统一标注数量

---

### 3.4 SSE事件 API (events.py)

#### GET /api/events/{channel}
SSE事件流
- **参数**: channel (路径参数)
- **响应**: Server-Sent Events 流

---

## 4. 数据流

### 4.1 标注数据流
```
前端 → labeling.py → stats_manager → data_store → 文件系统
                     ↓
              unified_labels.json
              pre_labels.json
```

### 4.2 多轮聚类数据流
```
前端 → multi_clustering.py → MultiClusteringManager → CharPoolManager
       ↓                                              ↓
   hog_clusters.json                            unified_labels.json
   labels.json
```

## 5. 当前问题与重构建议

### 5.1 架构问题
1. **路由分散**: 图片相关API分散在 app.py 和 images.py
2. **职责不清**: app.py 包含大量业务逻辑
3. **重复代码**: 多个地方有类似的JSON加载/保存逻辑
4. **配置耦合**: 硬编码路径在多个位置

### 5.2 重构建议

#### 5.2.1 目录结构重组
```
server/
├── main.py                 # 应用入口
├── config.py              # 配置管理
├── api/                   # API层
│   ├── images/           # 图片API模块
│   ├── labeling/         # 标注API模块
│   └── clustering/       # 聚类API模块
├── services/             # 业务逻辑层
│   ├── clustering/       # 聚类服务
│   ├── annotation/       # 标注服务
│   └── stats/            # 统计服务
├── models/               # 数据模型
│   ├── request.py        # 请求模型
│   └── response.py       # 响应模型
├── core/                 # 核心模块
│   ├── data_store.py     # 数据存储抽象
│   └── feature提取等
└── utils/                # 工具函数
```

#### 5.2.2 依赖注入
使用FastAPI的Depends进行依赖注入，提高可测试性

#### 5.2.3 统一响应格式
```python
class ApiResponse(BaseModel):
    code: int
    msg: str
    data: Any = None
```

#### 5.2.4 配置集中管理
所有配置通过config.py统一管理，支持环境变量覆盖

---

## 6. 前端UI架构

### 6.1 前端项目结构

```
server/frontend/
├── src/
│   ├── main.js              # 应用入口
│   ├── App.vue              # 根组件
│   ├── router/index.js      # 路由配置
│   ├── views/               # 页面视图
│   │   ├── Dashboard.vue       # 仪表盘
│   │   ├── OCRLabelPage.vue    # OCR标注页面
│   │   ├── MultiClusteringPage.vue    # 多轮聚类页面
│   │   ├── MultiClusteringLabelPage.vue  # 聚类标注页面
│   │   ├── CharManagementPage.vue       # 字符管理页面
│   │   ├── ClusterLabelPage.vue        # 聚类标注页面（旧）
│   │   └── DataStatisticsPage.vue      # 数据统计页面
│   ├── components/          # 公共组件
│   │   ├── Header.vue           # 顶部导航
│   │   ├── StatsCard.vue        # 统计卡片
│   │   ├── ClusterTable.vue     # 聚类表格
│   │   └── CharImage.vue        # 字符图片组件
│   ├── api/                 # API调用层
│   │   ├── labeling.js          # 标注相关API
│   │   ├── clustering.js        # 聚类相关API
│   │   └── images.js            # 图片相关API
│   └── utils/               # 工具函数
│       └── request.js           # HTTP请求封装
├── public/                  # 静态资源
├── index.html               # HTML模板
├── package.json             # 依赖配置
├── vite.config.js           # Vite配置
└── tailwind.config.js       # Tailwind配置
```

### 6.2 页面功能说明

#### 6.2.1 Dashboard (仪表盘)
- **路径**: `/`
- **功能**: 
  - 显示整体标注进度统计
  - 数据概览（总字符数、已标注、待标注）
  - 快速入口导航

#### 6.2.2 OCRLabelPage (OCR标注页面)
- **路径**: `/ocr-label/{image_id}`
- **功能**:
  - 显示PDF页面图片
  - 标注文字行位置和颜色
  - 支持撤销/重做操作

#### 6.2.3 MultiClusteringPage (多轮聚类页面)
- **路径**: `/multi-clustering`
- **功能**:
  - 聚类轮次选择标签
  - 参数配置面板（聚类方法、参数调整）
  - 聚类列表表格展示
  - 启动新轮聚类按钮

#### 6.2.4 MultiClusteringLabelPage (聚类标注页面)
- **路径**: `/mc-label/{round}/{cluster_id}`
- **功能**:
  - 显示聚类内所有字符图片
  - 批量标注功能
  - 相似度排序展示
  - 标注结果预览

#### 6.2.5 CharManagementPage (字符管理页面)
- **路径**: `/char-management`
- **功能**:
  - 字符列表展示（分页、搜索）
  - 字符统计信息
  - 字符图片预览

#### 6.2.6 DataStatisticsPage (数据统计页面)
- **路径**: `/statistics`
- **功能**:
  - 标注进度统计图表
  - 字符分布统计
  - 聚类效果分析

### 6.3 核心组件

#### 6.3.1 Header组件
- **功能**: 顶部导航栏
- **包含**: Logo、导航菜单、用户信息

#### 6.3.2 StatsCard组件
- **功能**: 统计数据卡片
- **包含**: 数值、标题、图标

#### 6.3.3 ClusterTable组件
- **功能**: 聚类列表表格
- **包含**: 聚类ID、状态、字符数、操作按钮

#### 6.3.4 CharImage组件
- **功能**: 字符图片展示
- **包含**: 图片显示、标注状态、操作按钮

### 6.4 路由配置

| 路径 | 组件 | 说明 |
|------|------|------|
| `/` | Dashboard.vue | 仪表盘 |
| `/ocr-label/:id` | OCRLabelPage.vue | OCR标注 |
| `/multi-clustering` | MultiClusteringPage.vue | 多轮聚类管理 |
| `/mc-label/:round/:cluster_id` | MultiClusteringLabelPage.vue | 聚类标注 |
| `/char-management` | CharManagementPage.vue | 字符管理 |
| `/statistics` | DataStatisticsPage.vue | 数据统计 |
| `/cluster-label/:cluster_id` | ClusterLabelPage.vue | 旧聚类标注（待移除） |

### 6.5 前端API调用

前端通过封装的API模块与后端交互：

```javascript
// api/labeling.js
export async function getStats() {
  return request.get('/api/labeling/stats')
}

export async function confirmAnnotation(data) {
  return request.post('/api/labeling/confirm', data)
}

// api/clustering.js
export async function startNewRound(params) {
  return request.post('/api/mc/rounds', params)
}

export async function getClusters(roundNum) {
  return request.get(`/api/mc/rounds/${roundNum}/clusters`)
}
```

### 6.6 状态管理

前端使用Vue 3的响应式API进行状态管理：

- **ref**: 基本类型状态（数字、字符串）
- **reactive**: 对象类型状态
- **computed**: 计算属性
- **watch**: 状态监听

### 6.7 UI框架

- **Vue 3**: 前端框架
- **Ant Design Vue**: UI组件库
- **Tailwind CSS**: 样式框架
- **Vite**: 构建工具

---

## 7. 交互流程

### 7.1 多轮聚类标注流程

```
1. 用户访问 /multi-clustering
2. 选择聚类方法（HDBSCAN/KMeans）
3. 调整聚类参数
4. 点击"启动新轮聚类"
5. 后端执行聚类算法
6. 显示聚类结果列表
7. 点击"标注"打开新窗口 /mc-label/{round}/{cluster_id}
8. 在标注页面选择字符进行标注
9. 保存标注结果
10. 返回聚类列表页面，状态更新为"已标注"
```

### 7.2 字符标注流程

```
1. 用户访问 /char-management
2. 搜索或浏览字符列表
3. 点击字符查看详情
4. 查看字符图片列表
5. 确认或修改标注
6. 批量保存标注结果
```

### 7.3 OCR标注流程

```
1. 用户访问 /ocr-label/{image_id}
2. 查看PDF页面图片
3. 标注文字行位置和颜色
4. 保存标注
5. 跳转到下一张图片
```
