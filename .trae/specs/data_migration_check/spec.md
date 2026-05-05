# 数据迁移与硬编码路径检查 - 产品需求文档

## Overview
- **Summary**: 检查前端代码中是否存在硬编码的数据集路径，验证数据迁移是否完整，确保数据集切换功能正常工作。
- **Purpose**: 解决数据集切换后仍然显示旧数据的问题，确保多数据集功能正确运行。
- **Target Users**: 开发人员、测试人员

## Goals
- 识别并修复前端代码中所有硬编码的数据集路径
- 验证数据文件是否正确放置在对应数据集文件夹中
- 确保数据集切换后显示正确的数据

## Non-Goals (Out of Scope)
- 不涉及后端代码修改
- 不添加新功能
- 不修改数据库结构

## Background & Context
当前系统已实现多数据集支持，但用户反馈切换数据集后仍然显示旧数据（如pdf01），需要检查代码中是否存在未更新的硬编码路径。

## Functional Requirements
- **FR-1**: 前端所有API调用必须支持动态数据集参数
- **FR-2**: 数据集切换后应显示对应数据集的数据
- **FR-3**: 不存在硬编码的数据集路径

## Non-Functional Requirements
- **NFR-1**: 代码检查应覆盖所有前端Vue组件
- **NFR-2**: 检查结果应可验证、可追溯

## Constraints
- **Technical**: Vue3 + Vite 框架
- **Dependencies**: axios 用于API调用

## Assumptions
- 后端API已正确实现多数据集支持
- 数据集文件夹结构已正确创建

## Acceptance Criteria

### AC-1: 所有API调用支持动态数据集
- **Given**: 前端发起API请求
- **When**: 请求发送时
- **Then**: 请求URL应包含dataset参数或请求头
- **Verification**: `programmatic`

### AC-2: 数据集切换功能正常
- **Given**: 用户在首页选择不同数据集
- **When**: 点击进入标注页面
- **Then**: 显示对应数据集的标注数据
- **Verification**: `human-judgment`

### AC-3: 无硬编码路径
- **Given**: 检查前端代码
- **When**: 搜索硬编码路径
- **Then**: 不应存在硬编码的数据集路径（如/pdf01/）
- **Verification**: `programmatic`

## Open Questions
- [ ] 是否所有前端页面都已更新支持数据集参数？
- [ ] 是否有遗漏的数据文件未迁移？