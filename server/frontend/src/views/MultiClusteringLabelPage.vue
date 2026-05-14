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
      <div class="toolbar-group">
        <span class="group-label">选择操作</span>
        <a-button @click="selectAll">全选</a-button>
        <a-button @click="clearSelection">清除选择</a-button>
      </div>
      
      <div class="toolbar-divider"></div>
      
      <div class="toolbar-group">
        <span class="group-label">批量标注</span>
        <a-input v-model:value="batchChar" placeholder="输入汉字" style="width: 80px;" />
        <a-button type="primary" @click="saveAll" :loading="saving" :disabled="!batchChar && selectedCount === 0">
          保存全部 ({{ selectedCount }})
        </a-button>
      </div>
      
      <div class="toolbar-divider"></div>
      
      <div class="toolbar-group">
        <span class="group-label">批量跳过</span>
        <a-button @click="batchSkip" danger :loading="batchSkipping" :disabled="selectedCount === 0">
          批量跳过 ({{ selectedCount }})
        </a-button>
        <a-button @click="batchUnskip" type="primary" :loading="batchUnskipping" :disabled="selectedCount === 0">
          批量撤回 ({{ selectedCount }})
        </a-button>
      </div>
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
            selected: selectedIndices.includes(img.index),
            skipped: img.char_status === 'skipped'
          }"
          @click="toggleSelect(img.index, $event)"
          @contextmenu.prevent="showLineContext(img)"
        >
          <div class="image-wrapper">
            <img :src="getImageUrl(img.char_id)" :alt="img.char_id" />
            <button v-if="img.char_status === 'skipped'" class="unskip-btn" @click.stop="unskipChar(img)">撤回</button>
            <button v-else class="skip-btn" @click.stop="skipChar(img)">跳过</button>
            <span v-if="img.char_status === 'labeled'" class="status-tag labeled-tag">已标注</span>
            <span v-else-if="img.char_status === 'skipped'" class="status-tag skipped-tag">已跳过</span>
            <div v-if="img.predicted_char" class="ocr-info" @click.stop>
              <a class="ocr-char-link" @click.prevent="goToPrelabel(img.predicted_char)">{{ img.predicted_char }}<span class="link-icon">↗</span></a>
              <span v-if="img.confidence" class="ocr-confidence" :class="img.confidence_level">{{ (img.confidence * 100).toFixed(0) }}%</span>
            </div>
          </div>
          <div class="label-area">
            <a-input
              v-if="img.char_status !== 'skipped'"
              :value="pendingLabels[img.index] || img.label || ''"
              placeholder="标注"
              style="width: 60px;"
              @input="updatePendingLabel(img.index, $event)"
              @blur="saveLabel(img)"
              @click.stop
            />
            <span v-else class="skipped-label">已跳过</span>
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
      <span>已跳过: {{ skippedCount }}</span>
      <span>未标注: {{ images.length - labeledCount - skippedCount }}</span>
      <span>已选: {{ selectedCount }}</span>
      <span>标注率: {{ images.length > 0 ? ((labeledCount / images.length) * 100).toFixed(1) : 0 }}%</span>
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
const batchSkipping = ref(false)
const batchUnskipping = ref(false)
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
const skippedCount = computed(() => images.value.filter(img => img.char_status === 'skipped').length)
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
        char_status: char.char_status || 'unlabeled',
        predicted_char: char.predicted_char || null,
        confidence: char.confidence || null,
        confidence_level: char.confidence_level || null,
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
  if (!batchChar.value || selectedIndices.value.length === 0) return

  saving.value = true
  try {
    // 使用批量提交接口
    const labels = selectedIndices.value.map(index => ({
      charIndex: index,
      char: batchChar.value
    }))

    await axios.post(`/api/mc/rounds/${round.value}/clusters/${clusterId.value}/labels/batch`, {
      labels
    })

    for (const index of selectedIndices.value) {
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

const skipChar = async (img) => {
  if (!confirm(`确定要跳过字符 ${img.char_id} 吗？跳过的字符将不再参与后续聚类。`)) return

  try {
    await axios.post(`/api/mc/rounds/${round.value}/chars/${img.char_id}/skip`)
    images.value.splice(img.index, 1)
    images.value = images.value.map((img, idx) => ({ ...img, index: idx }))
    displayImages.value = [...displayImages.value].filter(i => i.char_id !== img.char_id)
  } catch (err) {
    console.error('跳过字符失败:', err)
    alert('跳过字符失败: ' + (err.response?.data?.detail || err.message))
  }
}

const unskipChar = async (img) => {
  if (!confirm(`确定要撤回跳过字符 ${img.char_id} 吗？该字符将恢复为可标注状态。`)) return

  try {
    await axios.post(`/api/mc/rounds/${round.value}/chars/${img.char_id}/unskip`)
    images.value[img.index] = { ...img, char_status: 'unlabeled' }
  } catch (err) {
    console.error('撤回跳过失败:', err)
    alert('撤回跳过失败: ' + (err.response?.data?.detail || err.message))
  }
}

const batchSkip = async () => {
  if (selectedIndices.value.length === 0) return
  if (!confirm(`确定要批量跳过 ${selectedIndices.value.length} 个字符吗？跳过的字符将不再参与后续聚类。`)) return

  batchSkipping.value = true
  try {
    const charIds = selectedIndices.value
      .map(index => images.value[index]?.char_id)
      .filter(id => !!id)

    if (charIds.length === 0) return

    const res = await axios.post(`/api/mc/rounds/${round.value}/chars/batch-skip`, {
      char_ids: charIds
    })

    if (res.data.code === 0) {
      const skippedIds = new Set(charIds)
      images.value = images.value.map((img, idx) => {
        if (skippedIds.has(img.char_id)) {
          return { ...img, char_status: 'skipped', index: idx }
        }
        return { ...img, index: idx }
      })
      selectedIndices.value = []
    }
  } catch (err) {
    console.error('批量跳过失败:', err)
    alert('批量跳过失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    batchSkipping.value = false
  }
}

const batchUnskip = async () => {
  if (selectedIndices.value.length === 0) return
  if (!confirm(`确定要批量撤回跳过 ${selectedIndices.value.length} 个字符吗？这些字符将恢复为可标注状态。`)) return

  batchUnskipping.value = true
  try {
    const charIds = selectedIndices.value
      .map(index => images.value[index]?.char_id)
      .filter(id => !!id)

    if (charIds.length === 0) return

    const res = await axios.post(`/api/mc/rounds/${round.value}/chars/batch-unskip`, {
      char_ids: charIds
    })

    if (res.data.code === 0) {
      const unskippedIds = new Set(charIds)
      images.value = images.value.map(img => {
        if (unskippedIds.has(img.char_id)) {
          return { ...img, char_status: 'unlabeled' }
        }
        return img
      })
      selectedIndices.value = []
    }
  } catch (err) {
    console.error('批量撤回跳过失败:', err)
    alert('批量撤回跳过失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    batchUnskipping.value = false
  }
}

const goBack = () => {
  router.push('/multi-clustering')
}

const goToPrelabel = (char) => {
  const routeData = router.resolve({ path: `/prelabel-confirm/${encodeURIComponent(char)}` })
  window.open(routeData.href, '_blank')
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

.toolbar-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.group-label {
  font-size: 13px;
  color: #666;
  font-weight: 500;
}

.toolbar-divider {
  width: 1px;
  height: 24px;
  background: #d9d9d9;
}

.status-tag {
  position: absolute;
  bottom: 4px;
  left: 4px;
  padding: 1px 4px;
  font-size: 9px;
  font-weight: bold;
  border-radius: 2px;
}

.labeled-tag {
  background: rgba(82, 196, 26, 0.9);
  color: white;
}

.skipped-tag {
  background: rgba(255, 77, 79, 0.9);
  color: white;
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

.image-card:hover,
.image-card.skipped:hover,
.image-card.labeled:hover,
.image-card.pending:hover {
  border-color: #1890ff !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(24, 144, 255, 0.25);
}

.image-card.selected {
  border: 3px dashed #1890ff !important;
  background: #e6f7ff !important;
  box-shadow: 0 0 12px rgba(24, 144, 255, 0.4);
  transform: scale(1.02);
}

.image-card.labeled {
  border-color: #52c41a;
  background: #f6ffed;
}

.image-card.pending {
  border-color: #faad14;
  background: #fffbe6;
}

.image-card.skipped {
  border-color: #d9d9d9;
  background: #fafafa;
  opacity: 0.7;
}

.image-card.skipped.selected {
  opacity: 1;
  background: #e6f7ff !important;
  border: 3px dashed #1890ff !important;
  box-shadow: 0 0 12px rgba(24, 144, 255, 0.4);
  transform: scale(1.02);
}

.image-wrapper {
  position: relative;
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

.ocr-info {
  position: absolute;
  top: 4px;
  left: 4px;
  display: flex;
  align-items: center;
  gap: 2px;
  z-index: 5;
}

.ocr-char-link {
  font-size: 14px;
  font-weight: bold;
  color: #1890ff;
  background: rgba(255, 255, 255, 0.85);
  padding: 1px 4px;
  border-radius: 3px;
  line-height: 1.2;
  cursor: pointer;
  text-decoration: none;
  border-bottom: 1px dashed #1890ff;
}

.ocr-char-link:hover {
  color: #40a9ff;
  background: rgba(230, 247, 255, 0.95);
}

.ocr-char-link .link-icon {
  font-size: 10px;
  margin-left: 1px;
  opacity: 0;
  transition: opacity 0.2s;
}

.ocr-char-link:hover .link-icon {
  opacity: 1;
}

.ocr-confidence {
  font-size: 9px;
  padding: 1px 3px;
  border-radius: 3px;
  font-weight: 500;
  line-height: 1.2;
}

.ocr-confidence.high {
  background: rgba(82, 196, 26, 0.15);
  color: #52c41a;
}

.ocr-confidence.medium {
  background: rgba(250, 173, 20, 0.15);
  color: #faad14;
}

.ocr-confidence.low {
  background: rgba(255, 77, 79, 0.15);
  color: #ff4d4f;
}

.image-wrapper .skip-btn {
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 2px 6px;
  font-size: 10px;
  background: rgba(255, 77, 79, 0.9);
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s;
}

.image-wrapper:hover .skip-btn {
  opacity: 1;
}

.image-wrapper .skip-btn:hover {
  background: #ff4d4f;
}

.image-wrapper .unskip-btn {
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 2px 6px;
  font-size: 10px;
  background: rgba(82, 196, 26, 0.9);
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s;
}

.image-wrapper:hover .unskip-btn {
  opacity: 1;
}

.image-wrapper .unskip-btn:hover {
  background: #52c41a;
}

.image-wrapper .skipped-badge {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  padding: 4px 10px;
  font-size: 12px;
  font-weight: bold;
  background: rgba(255, 77, 79, 0.85);
  color: white;
  border-radius: 4px;
  pointer-events: none;
}

.skipped-label {
  color: #999;
  font-size: 12px;
  text-align: center;
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
