<template>
  <div class="line-check-page">
    <div class="page-header">
      <a-button @click="$router.push('/')" type="default">返回主页</a-button>
      <h1>行标注检查</h1>
    </div>

    <div class="stats-bar">
      <a-statistic title="总行数" :value="totalLines" />
      <a-statistic title="完全标注" :value="fullyLabeledCount" />
      <a-statistic title="部分标注" :value="partiallyLabeledCount" />
      <a-statistic title="未标注" :value="unlabeledCount" />
    </div>

    <div class="filter-bar">
      <a-select 
        v-model="filterStatus" 
        style="width: 150px;"
        @change="handleFilterChange"
      >
        <a-select-option value="all">全部</a-select-option>
        <a-select-option value="partial">部分标注</a-select-option>
        <a-select-option value="unlabeled">完全未标注</a-select-option>
        <a-select-option value="full">完全标注</a-select-option>
      </a-select>
    </div>

    <div class="line-list">
      <div 
        v-for="line in filteredLines" 
        :key="line.line_name"
        class="line-item"
        @click="goToLineDetail(line.line_name)"
      >
        <div class="line-info">
          <span class="line-name">{{ line.line_name }}</span>
          <span class="line-status" :class="line.status">
            {{ line.status === 'full' ? '✓' : line.status === 'partial' ? '~' : '?' }}
          </span>
        </div>
        <div class="progress-bar">
          <div 
            class="progress-fill" 
            :style="{ width: line.ratio + '%' }"
            :class="line.status"
          ></div>
        </div>
        <div class="line-stats">
          <span>{{ line.labeled }}/{{ line.total }}</span>
          <span class="ratio-text">{{ (line.ratio * 100).toFixed(1) }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'

const router = useRouter()
const lines = ref([])
const filterStatus = ref('all')

const totalLines = computed(() => lines.value.length)
const fullyLabeledCount = computed(() => lines.value.filter(l => l.status === 'full').length)
const partiallyLabeledCount = computed(() => lines.value.filter(l => l.status === 'partial').length)
const unlabeledCount = computed(() => lines.value.filter(l => l.status === 'unlabeled').length)

const filteredLines = computed(() => {
  if (filterStatus.value === 'all') return lines.value
  return lines.value.filter(l => l.status === filterStatus.value)
})

const loadLines = async () => {
  try {
    const response = await fetch('/api/line-status')
    const result = await response.json()
    if (result.code === 0) {
      lines.value = result.data
    } else {
      message.error('加载行数据失败')
    }
  } catch (error) {
    message.error('加载失败：' + error.message)
  }
}

const handleFilterChange = () => {}

const goToLineDetail = (lineName) => {
  router.push(`/line-detail/${lineName}`)
}

onMounted(() => {
  loadLines()
})
</script>

<style scoped>
.line-check-page {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
}

.page-header h1 {
  margin: 0;
  font-size: 24px;
}

.stats-bar {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
}

.filter-bar {
  margin-bottom: 20px;
}

.line-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.line-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 16px;
  background: #fafafa;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.line-item:hover {
  background: #e6f7ff;
}

.line-info {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 200px;
}

.line-name {
  font-size: 14px;
  font-weight: 500;
}

.line-status {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
}

.line-status.full {
  background: #95de64;
  color: #1890ff;
}

.line-status.partial {
  background: #ffe58f;
  color: #fa8c16;
}

.line-status.unlabeled {
  background: #f5f5f5;
  color: #d9d9d9;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: #e8e8e8;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
}

.progress-fill.full {
  background: #95de64;
}

.progress-fill.partial {
  background: #ffe58f;
}

.progress-fill.unlabeled {
  background: #d9d9d9;
}

.line-stats {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 120px;
  text-align: right;
}

.ratio-text {
  color: #999;
  font-size: 13px;
}

.line-detail {
  padding: 16px;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.detail-title {
  font-size: 18px;
  font-weight: bold;
}

.detail-progress {
  font-size: 14px;
  color: #1890ff;
}

.line-image-section {
  margin-bottom: 20px;
}

.image-container {
  background: #fafafa;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}

.line-image {
  max-width: 100%;
  max-height: 150px;
  object-fit: contain;
  border-radius: 4px;
}

.chars-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.char-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px 16px;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  min-width: 80px;
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
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 4px;
}

.cluster-info {
  font-size: 12px;
  color: #999;
}
</style>