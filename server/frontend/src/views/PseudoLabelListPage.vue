<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack">← 返回主页</a-button>
      <h2 class="title">伪标签传播</h2>
      <a-button size="small" @click="clearAllCache">清除全部缓存</a-button>
    </div>

    <div class="description">
      <p>选择一个已标记的标签，查看该标签在未标记聚类中的推荐图片，并进行标注。</p>
    </div>

    <div class="stats-section">
      <div class="stat-item">
        <span class="stat-value">{{ stats.total_chars }}</span>
        <span class="stat-label">已标记汉字</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">{{ stats.total_clusters }}</span>
        <span class="stat-label">涉及聚类</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">{{ stats.total_images }}</span>
        <span class="stat-label">涉及图片</span>
      </div>
    </div>

    <div class="labels-grid">
      <div
        v-for="item in labels"
        :key="item.char"
        class="label-card"
        @click="selectLabel(item)"
      >
        <div class="label-char">{{ item.char }}</div>
        <div class="label-stats">
          <span>标记: {{ item.total_count }}张</span>
          <span>聚类: {{ item.cluster_count }}个</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const labels = ref([])
const stats = ref({ total_chars: 0, total_clusters: 0, total_images: 0 })

const goBack = () => {
  router.push('/')
}

const clearAllCache = async () => {
  try {
    await axios.delete('http://localhost:5000/api/pseudo-labels/cache')
    alert('缓存已清除')
  } catch (err) {
    console.error('清除缓存失败:', err)
  }
}

const loadLabels = async () => {
  try {
    const res = await axios.get('http://localhost:5000/api/pseudo-labels')
    if (res.data.code === 0) {
      labels.value = res.data.data || []
      if (res.data.stats) {
        stats.value = res.data.stats
      }
    }
  } catch (err) {
    console.error('加载标签失败:', err)
  }
}

const selectLabel = (item) => {
  router.push(`/pseudo-label/${encodeURIComponent(item.char)}`)
}

onMounted(() => {
  loadLabels()
})
</script>

<style scoped>
.page-container { padding: 20px; max-width: 1200px; margin: 0 auto; }
.header { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
.title { margin: 0; }
.description {
  background: #f5f5f5;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 20px;
}
.description p { margin: 0; color: #666; }
.stats-section {
  display: flex;
  gap: 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px 32px;
  border-radius: 12px;
  margin-bottom: 24px;
}
.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #fff;
}
.stat-label {
  font-size: 14px;
  color: rgba(255, 255, 255, 0.8);
  margin-top: 4px;
}
.labels-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 16px;
}
.label-card {
  background: #fff;
  border: 2px solid #e8e8e8;
  border-radius: 12px;
  padding: 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}
.label-card:hover {
  border-color: #1890ff;
  box-shadow: 0 4px 12px rgba(24, 144, 255, 0.2);
}
.label-char {
  font-size: 48px;
  font-weight: bold;
  margin-bottom: 12px;
}
.label-stats {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #666;
}
</style>
