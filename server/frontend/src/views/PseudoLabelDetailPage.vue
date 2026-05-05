<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack">← 返回标签列表</a-button>
      <h2 class="title">标签「{{ char }}」的推荐聚类</h2>
      <a-button size="small" @click="clearCache">清除缓存</a-button>
    </div>

    <div v-if="loading" class="loading">加载中...</div>

    <div v-else-if="clusters.length === 0" class="empty">
      暂无推荐聚类（所有聚类可能已标记完毕）
    </div>

    <div v-else class="clusters-list">
      <div
        v-for="cluster in clusters"
        :key="cluster.cluster_id"
        class="cluster-item"
      >
        <div class="cluster-header">
          <div class="cluster-info">
            <span class="cluster-id">聚类 {{ cluster.cluster_id }}</span>
            <span class="cluster-stats">
              推荐: {{ cluster.matched_count }}张 / 总计: {{ cluster.total_count }}张
            </span>
            <span class="cluster-similarity">
              平均相似度: {{ (cluster.avg_similarity * 100).toFixed(1) }}%
            </span>
          </div>
          <div class="cluster-actions">
            <a-button type="primary" size="small" @click="loadClusterImages(cluster.cluster_id)">
              加载图片
            </a-button>
            <a-button size="small" @click="editCluster(cluster.cluster_id)">
              修改标注
            </a-button>
          </div>
        </div>

        <div v-if="expandedCluster === cluster.cluster_id" class="cluster-images">
          <div class="images-toolbar">
            <a-button size="small" @click="selectAllImages">全选</a-button>
            <a-button size="small" @click="clearSelection">清除选择</a-button>
            <a-button
              type="primary"
              size="small"
              @click="batchLabel"
              :disabled="selectedImages.length === 0"
            >
              标注选中 ({{ selectedImages.length }})
            </a-button>
          </div>

          <div
            class="images-grid"
            ref="containerRef"
            @mousedown="startSelection"
            @mousemove="updateSelection"
            @mouseup="endSelection"
            @mouseleave="endSelection"
          >
            <div
              v-for="img in currentImages"
              :key="img.index"
              :data-index="img.index"
              class="image-card"
              :class="{ selected: selectedImages.includes(img.index) }"
              @click="toggleSelect(img.index)"
            >
              <div class="image-wrapper">
                <img :src="getImageUrl(img.char_id)" :alt="img.char_id" />
              </div>
              <div class="similarity-badge">{{ (img.similarity * 100).toFixed(0) }}%</div>
            </div>
            <div
              v-if="isSelecting"
              class="selection-box"
              :style="selectionBoxStyle"
            ></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'

const route = useRoute()
const router = useRouter()
const char = decodeURIComponent(route.params.char)

const loading = ref(false)
const clusters = ref([])
const expandedCluster = ref(null)
const currentImages = ref([])
const selectedImages = ref([])

const containerRef = ref(null)
const isSelecting = ref(false)
const selectionStart = ref({ x: 0, y: 0 })
const selectionEnd = ref({ x: 0, y: 0 })

const selectionBoxStyle = computed(() => {
  const left = Math.min(selectionStart.value.x, selectionEnd.value.x)
  const top = Math.min(selectionStart.value.y, selectionEnd.value.y)
  const width = Math.abs(selectionEnd.value.x - selectionStart.value.x)
  const height = Math.abs(selectionEnd.value.y - selectionStart.value.y)
  return {
    left: left + 'px',
    top: top + 'px',
    width: width + 'px',
    height: height + 'px'
  }
})

const goBack = () => {
  router.push('/pseudo-label')
}

const editCluster = (clusterId) => {
  router.push(`/ocr-edit/${clusterId}`)
}

const clearCache = async () => {
  try {
    await axios.delete(`/api/pseudo-labels/cache/${encodeURIComponent(char)}`)
    loadClusters()
  } catch (err) {
    console.error('清除缓存失败:', err)
  }
}

const loadClusters = async () => {
  loading.value = true
  try {
    const res = await axios.post('/api/pseudo-labels/clusters', { char: char })
    if (res.data.code === 0) {
      clusters.value = res.data.clusters || []
    }
  } catch (err) {
    console.error('加载聚类失败:', err)
  } finally {
    loading.value = false
  }
}

const loadClusterImages = async (clusterId) => {
  try {
    const res = await axios.post('/api/pseudo-labels/clusters/images', { 
      char: char, 
      cluster_id: clusterId 
    })
    if (res.data.code === 0) {
      currentImages.value = res.data.images || []
      expandedCluster.value = clusterId
      selectedImages.value = []
    }
  } catch (err) {
    console.error('加载图片失败:', err)
  }
}

const getImageUrl = (charId) => {
  return `/api/char-images/${charId}.png`
}

const toggleSelect = (index) => {
  const idx = selectedImages.value.indexOf(index)
  if (idx === -1) {
    selectedImages.value.push(index)
  } else {
    selectedImages.value.splice(idx, 1)
  }
}

const selectAllImages = () => {
  selectedImages.value = currentImages.value.map(img => img.index)
}

const clearSelection = () => {
  selectedImages.value = []
}

const batchLabel = async () => {
  if (selectedImages.value.length === 0) return

  try {
    await axios.post('/api/cluster-labels/batch-save', {
      clusterId: expandedCluster.value,
      labels: selectedImages.value.map(idx => ({
        char: char,
        charIndex: idx
      }))
    })

    alert('标注成功！')
    selectedImages.value = []
    loadClusters()
  } catch (err) {
    console.error('标注失败:', err)
    alert('标注失败')
  }
}

const startSelection = (event) => {
  if (event.target.closest('.image-card')) return

  isSelecting.value = true
  const rect = containerRef.value.getBoundingClientRect()
  selectionStart.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  }
  selectionEnd.value = { ...selectionStart.value }
}

const updateSelection = (event) => {
  if (!isSelecting.value) return

  const rect = containerRef.value.getBoundingClientRect()
  selectionEnd.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  }
}

const endSelection = (event) => {
  if (!isSelecting.value) return

  isSelecting.value = false

  const rect = containerRef.value.getBoundingClientRect()
  const boxLeft = Math.min(selectionStart.value.x, selectionEnd.value.x)
  const boxRight = Math.max(selectionStart.value.x, selectionEnd.value.x)
  const boxTop = Math.min(selectionStart.value.y, selectionEnd.value.y)
  const boxBottom = Math.max(selectionStart.value.y, selectionEnd.value.y)

  const cardElements = containerRef.value.querySelectorAll('.image-card')
  const newSelected = []

  cardElements.forEach((card) => {
    const cardRect = card.getBoundingClientRect()
    const cardLeft = cardRect.left - rect.left
    const cardRight = cardRect.right - rect.left
    const cardTop = cardRect.top - rect.top
    const cardBottom = cardRect.bottom - rect.top

    if (cardLeft >= boxLeft && cardRight <= boxRight &&
        cardTop >= boxTop && cardBottom <= boxBottom) {
      const index = parseInt(card.getAttribute('data-index'))
      if (!selectedImages.value.includes(index)) {
        newSelected.push(index)
      }
    }
  })

  selectedImages.value = [...selectedImages.value, ...newSelected]
}

onMounted(() => {
  loadClusters()
})
</script>

<style scoped>
.page-container { padding: 20px; max-width: 1200px; margin: 0 auto; }
.header { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
.title { margin: 0; }
.loading, .empty {
  text-align: center;
  padding: 40px;
  color: #999;
}
.clusters-list { display: flex; flex-direction: column; gap: 20px; }
.cluster-item {
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 12px;
  padding: 16px;
}
.cluster-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.cluster-info {
  display: flex;
  align-items: center;
  gap: 16px;
}
.cluster-actions {
  display: flex;
  gap: 8px;
}
.cluster-id { font-weight: bold; font-size: 16px; }
.cluster-stats { color: #666; }
.cluster-similarity { color: #1890ff; }
.cluster-images { margin-top: 16px; }
.images-toolbar {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}
.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
  gap: 12px;
  position: relative;
  min-height: 200px;
  padding: 16px;
  background: #fafafa;
  border-radius: 8px;
  user-select: none;
}
.image-card {
  background: #fff;
  border: 2px solid #e8e8e8;
  border-radius: 8px;
  padding: 8px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}
.image-card:hover { border-color: #1890ff; }
.image-card.selected {
  border: 3px dashed #1890ff;
  background: #e6f7ff;
}
.image-wrapper { width: 70px; height: 70px; margin: 0 auto 8px; }
.image-wrapper img { width: 100%; height: 100%; object-fit: contain; }
.similarity-badge {
  font-size: 12px;
  color: #1890ff;
  font-weight: bold;
}
.selection-box {
  position: absolute;
  border: 2px dashed #1890ff;
  background: rgba(24, 144, 255, 0.1);
  pointer-events: none;
  z-index: 100;
}
</style>
