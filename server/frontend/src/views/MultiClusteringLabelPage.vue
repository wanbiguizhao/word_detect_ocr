<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack">← 返回列表</a-button>
      <h2 class="title">第 {{ round }} 轮聚类 {{ clusterId }} 标注</h2>
      <div class="cluster-info">
        <span>汉字: </span>
        <span class="chars-display">{{ charsDisplay || '?' }}</span>
      </div>
      <a-button type="primary" @click="saveAll" :loading="saving" :disabled="!batchChar && Object.keys(pendingLabels).length === 0">保存全部</a-button>
      <a-button @click="skipCluster" :loading="skipping" danger>暂不标记</a-button>
    </div>

    <div class="selection-toolbar">
      <a-button @click="selectAll">全选</a-button>
      <span>批量标注: </span>
      <a-input v-model:value="batchChar" placeholder="输入汉字" style="width: 80px;" />
      <a-button type="primary" @click="saveAll" :loading="saving" :disabled="!batchChar && selectedCount === 0">
        保存全部 ({{ selectedCount }})
      </a-button>
      <a-button @click="clearSelection">清除选择</a-button>
    </div>

    <div class="images-container" ref="containerRef"
      @mousedown="startSelection"
      @mousemove="updateSelection"
      @mouseup="endSelection"
      @mouseleave="endSelection"
    >
      <div class="images-grid">
        <div
          v-for="img in displayImages"
          :key="img.index"
          :data-index="img.index"
          class="image-card"
          :class="{
            labeled: img.label && !pendingLabels[img.index],
            pending: !!pendingLabels[img.index],
            selected: selectedIndices.includes(img.index)
          }"
          @click="toggleSelect(img.index, $event)"
          @contextmenu.prevent="showLineContext(img)"
        >
          <div class="image-wrapper">
            <img :src="getImageUrl(img.char_id)" :alt="img.char_id" />
          </div>
          <div class="label-area">
            <a-input
              :value="pendingLabels[img.index] || img.label || ''"
              placeholder="标注"
              style="width: 60px;"
              @input="updatePendingLabel(img.index, $event)"
              @blur="saveLabel(img)"
              @click.stop
            />
          </div>
          <div class="char-id">{{ img.char_id }}</div>
        </div>
      </div>

      <div
        v-if="isSelecting"
        class="selection-box"
        :style="selectionBoxStyle"
      ></div>
    </div>

    <div class="stats-bar">
      <span>总计: {{ images.length }}</span>
      <span>已标注: {{ labeledCount }}</span>
      <span>未标注: {{ images.length - labeledCount }}</span>
      <span>已选: {{ selectedCount }}</span>
      <span>标注率: {{ ((labeledCount / images.length) * 100).toFixed(1) }}%</span>
    </div>

  <a-modal v-model:open="showLineModal" title="行上下文" :footer="null" width="420px">
    <div style="text-align: center;">
      <img :src="lineContextImage" alt="行上下文" style="max-width: 100%; border: 1px solid #ddd;" />
    </div>
  </a-modal>

  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'

const route = useRoute()
const router = useRouter()

const round = computed(() => route.params.round)
const clusterId = computed(() => route.params.cluster_id)
const images = ref([])
const pendingLabels = ref({})
const batchChar = ref('')
const charsDisplay = ref('')
const saving = ref(false)
const skipping = ref(false)
const showLineModal = ref(false)
const lineContextImage = ref('')

const containerRef = ref(null)
const isSelecting = ref(false)
const selectionStart = ref({ x: 0, y: 0 })
const selectionEnd = ref({ x: 0, y: 0 })
const selectedIndices = ref([])

const displayImages = computed(() => {
  const sorted = [...images.value].sort((a, b) => {
    const aHasLabel = !!a.label
    const bHasLabel = !!b.label
    
    if (aHasLabel && !bHasLabel) return 1
    if (!aHasLabel && bHasLabel) return -1
    
    if (aHasLabel && bHasLabel) {
      const labelCounts = {}
      images.value.forEach(img => {
        if (img.label) {
          labelCounts[img.label] = (labelCounts[img.label] || 0) + 1
        }
      })
      const aCount = labelCounts[a.label] || 0
      const bCount = labelCounts[b.label] || 0
      if (aCount !== bCount) {
        return aCount - bCount
      }
      return a.label.localeCompare(b.label)
    }
    
    return 0
  })

  if (selectedIndices.value.length > 0) {
    const selected = sorted.filter(img => selectedIndices.value.includes(img.index))
    const notSelected = sorted.filter(img => !selectedIndices.value.includes(img.index))
    return [...selected, ...notSelected]
  }

  return sorted
})

const labeledCount = computed(() => images.value.filter(img => img.label).length)
const selectedCount = computed(() => selectedIndices.value.length)

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

const getImageUrl = (charId) => {
  return `/api/image/pdf_chars/${charId}`
}

const showLineContext = async (img) => {
  if (!img.line_name) {
    alert('无法获取行信息：缺少行名称')
    return
  }

  const lineName = img.line_name
  const lineUrl = `/api/line-images/${encodeURIComponent(lineName)}`

  let charLeft = img.col_start
  let charRight = img.col_end

  const match = img.char_id.match(/page_(\d+)_line_(\d+)_char_(\d+)/)
  const charIndex = match ? parseInt(match[3]) : 0

  try {
    const response = await fetch(lineUrl)
    
    if (!response.ok) {
      alert(`无法获取行图片：HTTP ${response.status}`)
      return
    }
    
    const blob = await response.blob()
    const bitmap = await createImageBitmap(blob)

    if (charLeft === undefined) {
      const avgCharWidth = Math.floor(bitmap.width / 30)
      charLeft = charIndex * avgCharWidth
      charRight = charLeft + avgCharWidth
    }

    const extend = 50
    const cropLeft = Math.max(0, charLeft - extend)
    const cropRight = Math.min(bitmap.width, (charRight || charLeft + 30) + extend)
    const cropWidth = cropRight - cropLeft

    const canvas = document.createElement('canvas')
    canvas.width = cropWidth
    canvas.height = bitmap.height
    const ctx = canvas.getContext('2d')

    ctx.drawImage(bitmap, cropLeft, 0, cropWidth, bitmap.height, 0, 0, cropWidth, bitmap.height)

    ctx.strokeStyle = 'red'
    ctx.lineWidth = 2
    
    const leftLineX = charLeft - cropLeft
    ctx.beginPath()
    ctx.moveTo(leftLineX, 0)
    ctx.lineTo(leftLineX, bitmap.height)
    ctx.stroke()
    
    if (charRight !== undefined) {
      const rightLineX = charRight - cropLeft
      ctx.beginPath()
      ctx.moveTo(rightLineX, 0)
      ctx.lineTo(rightLineX, bitmap.height)
      ctx.stroke()
    }

    lineContextImage.value = canvas.toDataURL('image/png')
    showLineModal.value = true
  } catch (err) {
    console.error('加载行图片失败:', err)
    alert(`加载行图片失败：${err.message}`)
  }
}

const loadData = async () => {
  try {
    const res = await axios.get(`/api/mc/rounds/${round.value}/clusters/${clusterId.value}`)
    if (res.data.code === 0) {
      const data = res.data
      images.value = (data.chars || []).map((char, index) => ({
        index,
        char_id: char.char_id,
        label: char.label || '',
        line_name: char.line_name,
        col_start: char.col_start,
        col_end: char.col_end,
        lineage: {
          line_name: char.line_name,
          col_start: char.col_start,
          col_end: char.col_end
        }
      }))

      charsDisplay.value = data.char || ''
    }
  } catch (err) {
    console.error('加载数据失败:', err)
  }
}

const toggleSelect = (index, event) => {
  if (event && event.ctrlKey) {
    const idx = selectedIndices.value.indexOf(index)
    if (idx === -1) {
      selectedIndices.value.push(index)
    } else {
      selectedIndices.value.splice(idx, 1)
    }
  } else {
    const idx = selectedIndices.value.indexOf(index)
    if (idx === -1) {
      selectedIndices.value.push(index)
    } else {
      selectedIndices.value.splice(idx, 1)
    }
  }
}

const selectAll = () => {
  selectedIndices.value = images.value.map((_, i) => i)
}

const clearSelection = () => {
  selectedIndices.value = []
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
      if (!selectedIndices.value.includes(index)) {
        newSelected.push(index)
      }
    }
  })

  selectedIndices.value = [...selectedIndices.value, ...newSelected]
}

const updatePendingLabel = (index, event) => {
  pendingLabels.value[index] = event.target.value
}

const saveLabel = async (img) => {
  const label = pendingLabels.value[img.index]
  if (!label) return

  try {
    await axios.post(`/api/mc/rounds/${round.value}/clusters/${clusterId.value}/labels`, {
      charIndex: img.index,
      char: label
    })
    img.label = label
    delete pendingLabels.value[img.index]
  } catch (err) {
    console.error('保存标签失败:', err)
  }
}

const saveAll = async () => {
  if (!batchChar.value) return

  saving.value = true
  try {
    for (const index of selectedIndices.value) {
      await axios.post(`/api/mc/rounds/${round.value}/clusters/${clusterId.value}/labels`, {
        charIndex: index,
        char: batchChar.value
      })
      const img = images.value[index]
      if (img) {
        img.label = batchChar.value
      }
    }
    pendingLabels.value = {}
    selectedIndices.value = []
    batchChar.value = ''
    await loadData()
  } catch (err) {
    console.error('批量保存失败:', err)
  } finally {
    saving.value = false
  }
}

const skipCluster = async () => {
  if (!confirm('确定要跳过这个聚类吗？')) return

  skipping.value = true
  try {
    await axios.post(`/api/mc/rounds/${round.value}/clusters/${clusterId.value}/skip`)
    router.push('/multi-clustering')
  } catch (err) {
    console.error('跳过聚类失败:', err)
  } finally {
    skipping.value = false
  }
}

const goBack = () => {
  router.push('/multi-clustering')
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.page-container {
  padding: 20px;
}

.header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.title {
  margin: 0;
  font-size: 18px;
}

.cluster-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.selection-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  padding: 12px;
  background: #f5f5f5;
  border-radius: 4px;
}

.images-container {
  position: relative;
  min-height: 400px;
}

.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
}

.image-card {
  border: 2px solid #e8e8e8;
  border-radius: 8px;
  padding: 8px;
  background: #fff;
  transition: all 0.2s;
}

.image-card:hover {
  border-color: #1890ff;
}

.image-card.selected {
  border-color: #1890ff;
  background: #e6f7ff;
}

.image-card.labeled {
  border-color: #52c41a;
  background: #f6ffed;
}

.image-card.pending {
  border-color: #faad14;
  background: #fffbe6;
}

.image-wrapper {
  width: 100%;
  aspect-ratio: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #fafafa;
  border-radius: 4px;
  overflow: hidden;
}

.image-wrapper img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.label-area {
  margin-top: 8px;
  text-align: center;
}

.char-id {
  margin-top: 4px;
  font-size: 10px;
  color: #999;
  text-align: center;
  word-break: break-all;
}

.selection-box {
  position: absolute;
  border: 2px dashed #1890ff;
  background: rgba(24, 144, 255, 0.1);
  pointer-events: none;
}

.stats-bar {
  display: flex;
  gap: 24px;
  padding: 12px;
  margin-top: 16px;
  background: #fafafa;
  border-radius: 4px;
  font-size: 14px;
}

.stats-bar span {
  color: #666;
}

.chars-display {
  font-size: 18px;
  font-weight: bold;
  color: #1890ff;
}
</style>
