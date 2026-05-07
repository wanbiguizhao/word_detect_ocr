<template>
  <div class="home-container">
    <div class="header">
      <h1 class="title">汉字识别标注系统</h1>
      <p class="subtitle">多功能标注平台</p>
    </div>

    <div class="feature-grid">
      <div class="feature-card" @click="goTo('/segment')">
        <div class="feature-icon">
          <span class="icon">✂️</span>
        </div>
        <h2 class="feature-title">字符切割标注</h2>
        <p class="feature-desc">
          对汉字图片进行切割标注，支持主动学习样本筛选
        </p>
        <div class="feature-tags">
          <a-tag color="blue">主动学习</a-tag>
          <a-tag color="green">样本筛选</a-tag>
        </div>
      </div>

      <div class="feature-card" @click="goTo('/ocr')">
        <div class="feature-icon">
          <span class="icon">📝</span>
        </div>
        <h2 class="feature-title">汉字OCR标注</h2>
        <p class="feature-desc">
          对聚类后的汉字进行OCR标注，支持混乱度排序和批量标注
        </p>
        <div class="feature-tags">
          <a-tag color="purple">聚类标注</a-tag>
          <a-tag color="orange">混乱度分析</a-tag>
        </div>
      </div>

      <div class="feature-card" @click="goTo('/pseudo-label')">
        <div class="feature-icon">
          <span class="icon">🔍</span>
        </div>
        <h2 class="feature-title">伪标签传播</h2>
        <p class="feature-desc">
          利用已标记的数据，对未标记的聚类进行标签推荐和批量标注
        </p>
        <div class="feature-tags">
          <a-tag color="cyan">半监督</a-tag>
          <a-tag color="blue">标签传播</a-tag>
        </div>
      </div>

      <div class="feature-card" @click="goTo('/char-management')">
        <div class="feature-icon">
          <span class="icon">📋</span>
        </div>
        <h2 class="feature-title">汉字管理</h2>
        <p class="feature-desc">
          管理已标注汉字，查看详情和标注情况
        </p>
        <div class="feature-tags">
          <a-tag color="green">数据统计</a-tag>
          <a-tag color="blue">汉字详情</a-tag>
        </div>
      </div>

      <div class="feature-card" @click="goTo('/line-check')">
        <div class="feature-icon">
          <span class="icon">📊</span>
        </div>
        <h2 class="feature-title">行标注检查</h2>
        <p class="feature-desc">
          检查哪些行还未完全标注，快速定位缺失数据
        </p>
        <div class="feature-tags">
          <a-tag color="orange">进度检查</a-tag>
          <a-tag color="red">缺失定位</a-tag>
        </div>
      </div>

      <div class="feature-card" @click="goTo('/migration-list')">
        <div class="feature-icon">
          <span class="icon">🚀</span>
        </div>
        <h2 class="feature-title">汉字OCR标注迁移</h2>
        <p class="feature-desc">
          基于视觉特征匹配，利用已有标注数据快速标注新数据集
        </p>
        <div class="feature-tags">
          <a-tag color="purple">迁移学习</a-tag>
          <a-tag color="green">智能推荐</a-tag>
        </div>
      </div>
    </div>

    <div class="footer">
      <p>数据来源：word_detect_ocr</p>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const goTo = (path) => {
  router.push(path)
}
</script>

<style scoped>
.home-container {
  padding: 40px;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: 50px;
}

.title {
  font-size: 36px;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 12px;
}

.subtitle {
  font-size: 18px;
  color: #666;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
  margin-bottom: 40px;
}

.feature-card {
  background: #fff;
  border-radius: 12px;
  padding: 32px;
  cursor: pointer;
  transition: all 0.3s ease;
  border: 2px solid #e8e8e8;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.feature-card:hover:not(.disabled) {
  border-color: #1890ff;
  box-shadow: 0 4px 16px rgba(24, 144, 255, 0.15);
  transform: translateY(-4px);
}

.feature-card.disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.feature-icon {
  margin-bottom: 16px;
}

.icon {
  font-size: 48px;
}

.feature-title {
  font-size: 22px;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 12px;
}

.feature-desc {
  font-size: 14px;
  color: #666;
  line-height: 1.6;
  margin-bottom: 16px;
}

.feature-tags {
  display: flex;
  gap: 8px;
}

.footer {
  text-align: center;
  color: #999;
  font-size: 14px;
  padding-top: 20px;
  border-top: 1px solid #e8e8e8;
}

@media (max-width: 768px) {
  .feature-grid {
    grid-template-columns: 1fr;
  }
}
</style>