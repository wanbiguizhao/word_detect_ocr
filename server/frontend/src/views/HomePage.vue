<template>
  <div class="home-container">
    <div class="header">
      <h1 class="title">汉字识别标注系统</h1>
      <p class="subtitle">多功能标注平台</p>
    </div>

    <!-- 分类区域 -->
    
    <!-- 字符切割标注 -->
    <div class="category-section">
      <div class="category-header">
        <span class="category-icon">✂️</span>
        <h2 class="category-title">字符切割标注</h2>
        <p class="category-desc">对PDF文档进行字符级切割和标注</p>
      </div>
      <div class="category-grid">
        <div class="feature-card" @click="goTo('/segment')">
          <div class="feature-icon">
            <span class="icon">✂️</span>
          </div>
          <h3 class="feature-title">字符切割标注</h3>
          <p class="feature-desc">
            对汉字图片进行切割标注，支持主动学习样本筛选
          </p>
          <div class="feature-tags">
            <a-tag color="blue">主动学习</a-tag>
            <a-tag color="green">样本筛选</a-tag>
          </div>
        </div>

        <div class="feature-card" @click="goTo('/line-check')">
          <div class="feature-icon">
            <span class="icon">📊</span>
          </div>
          <h3 class="feature-title">行标注检查</h3>
          <p class="feature-desc">
            检查哪些行还未完全标注，快速定位缺失数据
          </p>
          <div class="feature-tags">
            <a-tag color="orange">进度检查</a-tag>
            <a-tag color="red">缺失定位</a-tag>
          </div>
        </div>
      </div>
    </div>

    <!-- 聚类标注 -->
    <div class="category-section">
      <div class="category-header">
        <span class="category-icon">🧩</span>
        <h2 class="category-title">聚类标注</h2>
        <p class="category-desc">对相似字符进行聚类，批量标注</p>
      </div>
      <div class="category-grid">
        <div class="feature-card" @click="goTo('/ocr')">
          <div class="feature-icon">
            <span class="icon">📝</span>
          </div>
          <h3 class="feature-title">汉字OCR标注</h3>
          <p class="feature-desc">
            对聚类后的汉字进行OCR标注，支持混乱度排序和批量标注
          </p>
          <div class="feature-tags">
            <a-tag color="purple">聚类标注</a-tag>
            <a-tag color="orange">混乱度分析</a-tag>
          </div>
        </div>

        <div class="feature-card" @click="goTo('/multi-clustering')">
          <div class="feature-icon">
            <span class="icon">🔄</span>
          </div>
          <h3 class="feature-title">多轮聚类标注</h3>
          <p class="feature-desc">
            支持多次聚类，自动排除已标注字符，逐步完善标注
          </p>
          <div class="feature-tags">
            <a-tag color="cyan">多轮迭代</a-tag>
            <a-tag color="blue">智能排除</a-tag>
          </div>
        </div>
      </div>
    </div>

    <!-- OCR预标注 -->
    <div class="category-section">
      <div class="category-header">
        <span class="category-icon">🤖</span>
        <h2 class="category-title">OCR预标注</h2>
        <p class="category-desc">利用模型和迁移学习进行智能预标注</p>
      </div>
      <div class="category-grid">
        <div class="feature-card" @click="goTo('/dashboard')">
          <div class="feature-icon">
            <span class="icon">🎯</span>
          </div>
          <h3 class="feature-title">标注仪表盘</h3>
          <p class="feature-desc">
            查看标注进度，生成预标注，管理标注任务
          </p>
          <div class="feature-tags">
            <a-tag color="cyan">进度监控</a-tag>
            <a-tag color="purple">智能标注</a-tag>
          </div>
        </div>

      </div>
    </div>

    <div class="footer">
      <p>数据来源：word_detect_ocr</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const goTo = (path) => {
  router.push(path)
}

onMounted(() => {
})
</script>

<style scoped>
.home-container {
  padding: 40px;
  max-width: 1400px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: 40px;
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

/* Category Section */
.category-section {
  margin-bottom: 40px;
}

.category-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 2px solid #e8e8e8;
}

.category-icon {
  font-size: 28px;
}

.category-title {
  font-size: 24px;
  font-weight: 600;
  color: #1a1a1a;
  margin: 0;
}

.category-desc {
  font-size: 14px;
  color: #666;
  margin-left: auto;
}

.category-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
}

/* Feature Card */
.feature-card {
  background: #fff;
  border-radius: 12px;
  padding: 28px;
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
  margin-bottom: 14px;
}

.icon {
  font-size: 40px;
}

.feature-title {
  font-size: 20px;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 10px;
}

.feature-desc {
  font-size: 14px;
  color: #666;
  line-height: 1.6;
  margin-bottom: 14px;
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
  margin-top: 20px;
}

@media (max-width: 768px) {
  .home-container {
    padding: 20px;
  }

  .category-grid {
    grid-template-columns: 1fr;
  }

  .category-header {
    flex-wrap: wrap;
  }

  .category-desc {
    margin-left: 0;
    width: 100%;
  }
}
</style>