<template>
  <div class="line-detail-page">
    <div class="page-header">
      <a-button @click="$router.push('/line-check')" type="default">返回列表</a-button>
      <h1>{{ lineName }}</h1>
    </div>

    <div v-if="lineData" class="line-content">
      <div class="progress-info">
        <span class="progress-text">{{ lineData.labeled }}/{{ lineData.total }} 已标注</span>
        <a-progress 
          :percent="lineData.ratio * 100" 
          :show-info="false"
          stroke-color="#1890ff"
          style="width: 200px;"
        />
      </div>

      <div class="image-section">
        <h3>行图片</h3>
        <div class="image-container">
          <img 
            :src="`/api/line-images/${lineName}`" 
            :alt="lineName"
            class="line-image"
            @error="handleImageError"
          />
        </div>
      </div>

      <div class="chars-section">
        <h3>标注汉字</h3>
        <div class="chars-grid">
          <div 
            v-for="(charInfo, index) in lineData.chars" 
            :key="index"
            class="char-item"
            :class="{ 'unlabeled': !charInfo.labeled }"
            @click="goToCluster(charInfo)"
          >
            <span class="char-label">{{ charInfo.char || '?' }}</span>
            <span class="cluster-info">聚类 {{ charInfo.cluster_id }}</span>
          </div>
        </div>
      </div>
    </div>

    <div v-else class="loading">
      <a-spin size="large" tip="加载中..." />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { message } from 'ant-design-vue'

const route = useRoute()
const lineName = ref('')
const lineData = ref(null)

const loadLineData = async () => {
  try {
    const response = await fetch('/api/line-status')
    const result = await response.json()
    if (result.code === 0) {
      lineData.value = result.data.find(l => l.line_name === lineName.value)
      if (!lineData.value) {
        message.error('未找到该行数据')
      }
    } else {
      message.error('加载失败')
    }
  } catch (error) {
    message.error('加载失败：' + error.message)
  }
}

const goToCluster = (charInfo) => {
  if (charInfo.cluster_id) {
    window.open(`/ocr-label/${charInfo.cluster_id}`, '_blank')
  }
}

const handleImageError = (e) => {
  e.target.src = ''
  e.target.style.display = 'none'
}

onMounted(() => {
  lineName.value = route.params.lineName
  loadLineData()
})
</script>

<style scoped>
.line-detail-page {
  padding: 20px;
  max-width: 1000px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.page-header h1 {
  margin: 0;
  font-size: 24px;
}

.line-content {
  background: #fff;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.progress-info {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #e8e8e8;
}

.progress-text {
  font-size: 16px;
  font-weight: 500;
  color: #1890ff;
}

.image-section {
  margin-bottom: 24px;
  overflow-x: auto;
}

.image-section h3 {
  font-size: 16px;
  margin-bottom: 12px;
  color: #333;
}

.image-container {
  background: #fafafa;
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  min-width: max-content;
  width: fit-content;
}

.line-image {
  width: auto;
  height: auto;
  max-height: 150px;
  border-radius: 4px;
}

.chars-section {
  padding-top: 16px;
  border-top: 1px solid #e8e8e8;
}

.chars-section h3 {
  font-size: 16px;
  margin-bottom: 16px;
  color: #333;
}

.chars-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding-bottom: 10px;
}

.char-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 40px;
  padding: 8px 4px;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.char-item:hover {
  border-color: #1890ff;
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.2);
}

.char-item.unlabeled {
  background: #fff7e6;
  border-color: #ffc53d;
}

.char-label {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 2px;
  white-space: nowrap;
}

.char-item.unlabeled .char-label {
  color: #fa8c16;
}

.cluster-info {
  font-size: 12px;
  color: #999;
}

.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
}
</style>