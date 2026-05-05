# 数据迁移与硬编码路径检查 - 实施计划

## [x] Task 1: 检查前端所有Vue组件中的硬编码路径
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 搜索所有.vue文件中是否存在硬编码的数据集路径（如/pdf01/）
  - 检查axios调用是否都使用了动态数据集参数
- **Acceptance Criteria Addressed**: AC-1, AC-3
- **Test Requirements**:
  - `programmatic` TR-1.1: 搜索结果不应包含硬编码的数据集路径
  - `programmatic` TR-1.2: 所有axios调用应包含dataset参数
- **Notes**: 使用grep工具搜索

## [x] Task 2: 检查数据集文件夹结构
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 验证各数据集文件夹是否存在必要的子目录
  - 检查标注数据文件是否正确放置
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-2.1: 每个数据集应包含clusters、labeling、pdf_chars、pdf_lines目录
  - `programmatic` TR-2.2: labels.json文件应存在于对应数据集目录
- **Notes**: 使用文件系统检查

## [x] Task 3: 检查首页数据集选择器
- **Priority**: P1
- **Depends On**: Task 1
- **Description**: 
  - 验证首页数据集选择器是否正确工作
  - 检查localStorage是否正确保存选择
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `human-judgment` TR-3.1: 数据集选择下拉框应显示所有可用数据集
  - `programmatic` TR-3.2: localStorage应正确存储selectedDataset
- **Notes**: 测试交互功能

## [x] Task 4: 检查标注页面API调用
- **Priority**: P0
- **Depends On**: Task 1
- **Description**: 
  - 验证OcrClusterLabelPage.vue等标注页面的API调用
  - 确保所有请求都携带dataset参数
- **Acceptance Criteria Addressed**: AC-1, AC-2
- **Test Requirements**:
  - `programmatic` TR-4.1: 所有axios调用应包含dataset参数
  - `human-judgment` TR-4.2: 切换数据集后显示对应数据
- **Notes**: 重点检查主要标注页面

## [x] Task 5: 修复发现的问题
- **Priority**: P0
- **Depends On**: Task 1, Task 2, Task 3, Task 4
- **Description**: 
  - 修复发现的硬编码路径问题
  - 确保数据文件正确放置
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-3
- **Test Requirements**:
  - `programmatic` TR-5.1: 修复后不应存在硬编码路径
  - `human-judgment` TR-5.2: 数据集切换功能正常工作
- **Notes**: 根据前序任务结果进行修复