<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack">← 返回汉字列表</a-button>
      <h2 class="title">预标注详情 - {{ char }}</h2>
      <div class="header-stats">
        <span>总数: {{ total }}</span>
        <span>已确认: {{ confirmed }}</span>
        <span class="highlight">待确认: {{ pending }}</span>
      </div>
      <div class="header-actions">
        <label class="select-all">
          <input 
            type="checkbox" 
            v-model="selectAll" 
            :disabled="pending === 0"
            @change="handleSelectAll"
          />
          <span>全选</span>
        </label>
        <a-button 
          type="primary" 
          :disabled="selectedItems.length === 0"
          @click="confirmBatch"
        >
          ✓ 批量确认 ({{ selectedItems.length }})
        </a-button>
        <a-button 
          :disabled="selectedItems.length === 0"
          @click="clearSelection"
        >
          ✗ 取消选择
        </a-button>
        <a-button 
          @click="selectByConfidence('high')"
        >
          🟢 选择高置信度
        </a-button>
        <a-button 
          @click="selectByConfidence('medium')"
        >
          🟠 选择中置信度
        </a-button>
        <a-button 
          @click="selectByConfidence('low')"
        >
          🔴 选择低置信度
        </a-button>
      </div>
    </div>

    <div v-if="loading" class="loading">加载中...</div>

    <div v-else-if="prelabels.length === 0" class="empty">
      暂无预标注数据
    </div>

    <div v-else class="prelabels-grid">
      <div
        v-for="(item, index) in prelabels"
        :key="item.char_id"
        class="prelabel-card"
        :class="{ 
          'high-confidence': item.confidence_level === 'high',
          'medium-confidence': item.confidence_level === 'medium',
          'low-confidence': item.confidence_level === 'low',
          'confirmed': item.status === 'confirmed',
          'selected': isSelected(item.char_id)
        }"
        @click="handleCardClick(item)"
        @contextmenu.prevent="showLineContext(item)"
      >
        <input 
          v-if="item.status !== 'confirmed'"
          type="checkbox" 
          class="select-checkbox"
          :checked="isSelected(item.char_id)"
          @change.stop="toggleSelect(item.char_id)"
        />
        <div class="image-wrapper">
          <img :src="getImageUrl(item.image_path)" :alt="item.char_id" />
        </div>
        <div class="info-row">
          <div class="status-tag">{{ item.status === 'confirmed' ? '已确认' : '待确认' }}</div>
          <div class="confidence-badge" :class="item.confidence_level">
            {{ (item.confidence * 100).toFixed(0) }}%
          </div>
        </div>
        <div class="predicted-char">{{ item.predicted_char }}</div>
        <div class="actions">
          <a-button size="small" @click="modifyLabel(index)">修改</a-button>
          <a-button size="small" @click="skipLabel(index)">跳过</a-button>
        </div>
      </div>
    </div>

    <div v-if="showModifyModal" class="modal-overlay" @click.self="closeModifyModal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>修改标注</h3>
          <a-button @click="closeModifyModal">×</a-button>
        </div>
        <div class="modal-body">
          <div class="modal-image">
            <img :src="getImageUrl(modifyItem?.image_path)" :alt="modifyItem?.char_id" />
          </div>
          <div class="current-label">
            当前预测: <span class="char">{{ modifyItem?.predicted_char }}</span>
            (置信度: {{ (modifyItem?.confidence * 100).toFixed(0) }}%)
          </div>
          <a-input 
            v-model="newLabel" 
            placeholder="请输入正确的汉字"
            class="input"
            @keyup.enter="saveModifiedLabel"
          />
        </div>
        <div class="modal-footer">
          <a-button @click="closeModifyModal">取消</a-button>
          <a-button type="primary" @click="saveModifiedLabel">保存</a-button>
        </div>
      </div>
    </div>

    <div v-if="showLineModal" class="modal-overlay" @click.self="closeLineModal">
      <div class="modal-content line-modal">
        <div class="modal-header">
          <h3>行上下文 - {{ currentLineChar }}</h3>
          <a-button @click="closeLineModal">×</a-button>
        </div>
        <div class="modal-body line-modal-body">
          <div style="text-align: center;">
            <img :src="lineContextImage" alt="行上下文" style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px;" />
          </div>
          <div class="line-modal-hint">
            红色线条标记了当前字符的边界（左右各扩展50像素）
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'

const route = useRoute()
const router = useRouter()
const char = decodeURIComponent(route.params.char)

const loading = ref(false)
const prelabels = ref([])
const showModifyModal = ref(false)
const modifyItem = ref(null)
const modifyIndex = ref(-1)
const newLabel = ref('')
const selectAll = ref(false)
const selectedItems = ref([])

// 行上下文相关状态
const showLineModal = ref(false)
const lineContextImage = ref('')
const currentLineChar = ref('')

// 框选相关状态
const isDrawing = ref(false)
const startPoint = ref({ x: 0, y: 0 })
const endPoint = ref({ x: 0, y: 0 })
const selectionRect = ref(null)
const cardRefs = ref([])

const total = computed(() => prelabels.value.length)
const confirmed = computed(() => prelabels.value.filter(p => p.status === 'confirmed').length)
const pending = computed(() => prelabels.value.filter(p => p.status !== 'confirmed').length)

const goBack = () => {
  router.push('/dashboard')
}

const getImageUrl = (imagePath) => {
  return `/api/char-images/${encodeURIComponent(imagePath)}`
}

const loadPreLabels = async () => {
  loading.value = true
  try {
    const res = await axios.get(`/api/labeling/prelabels/${encodeURIComponent(char)}`, {
      params: { page_size: 1000 }
    })
    if (res.data.code === 0) {
      // 排序逻辑：已确认的放在最后，未确认的按置信度升序排列
      prelabels.value = (res.data.data || []).sort((a, b) => {
        // 已确认的放在后面
        const aConfirmed = a.status === 'confirmed'
        const bConfirmed = b.status === 'confirmed'
        if (aConfirmed && !bConfirmed) return 1
        if (!aConfirmed && bConfirmed) return -1
        // 未确认的按置信度升序排列
        const confA = a.confidence || 0
        const confB = b.confidence || 0
        return confA - confB
      })
    }
  } catch (err) {
    console.error('加载预标注失败:', err)
  } finally {
    loading.value = false
  }
}

const isSelected = (charId) => {
  return selectedItems.value.includes(charId)
}

const toggleSelect = (charId) => {
  const index = selectedItems.value.indexOf(charId)
  if (index > -1) {
    selectedItems.value.splice(index, 1)
  } else {
    selectedItems.value.push(charId)
  }
  updateSelectAll()
}

const handleSelectAll = () => {
  if (selectAll.value) {
    // 全选待确认的项
    const pendingItems = prelabels.value.filter(p => p.status !== 'confirmed')
    selectedItems.value = pendingItems.map(p => p.char_id)
  } else {
    selectedItems.value = []
  }
}

const updateSelectAll = () => {
  const pendingItems = prelabels.value.filter(p => p.status !== 'confirmed')
  selectAll.value = pendingItems.length > 0 && 
    pendingItems.every(p => selectedItems.value.includes(p.char_id))
}

const clearSelection = () => {
  selectedItems.value = []
  selectAll.value = false
}

const handleCardClick = (item) => {
  if (item.status !== 'confirmed') {
    toggleSelect(item.char_id)
  }
}

const selectByConfidence = (level) => {
  const items = prelabels.value.filter(p => p.status !== 'confirmed' && p.confidence_level === level)
  items.forEach(item => {
    if (!selectedItems.value.includes(item.char_id)) {
      selectedItems.value.push(item.char_id)
    }
  })
  updateSelectAll()
}

// 框选功能
const handleMouseDown = (e) => {
  if (e.button !== 0) return
  isDrawing.value = true
  startPoint.value = { x: e.clientX, y: e.clientY }
  endPoint.value = { x: e.clientX, y: e.clientY }
  createSelectionRect()
}

const handleMouseMove = (e) => {
  if (!isDrawing.value) return
  endPoint.value = { x: e.clientX, y: e.clientY }
  updateSelectionRect()
}

const handleMouseUp = () => {
  if (!isDrawing.value) return
  isDrawing.value = false
  selectInRect()
  removeSelectionRect()
}

const createSelectionRect = () => {
  const rect = document.createElement('div')
  rect.className = 'selection-rect'
  document.body.appendChild(rect)
  selectionRect.value = rect
  updateSelectionRect()
}

const updateSelectionRect = () => {
  if (!selectionRect.value) return
  const rect = selectionRect.value
  const x = Math.min(startPoint.value.x, endPoint.value.x)
  const y = Math.min(startPoint.value.y, endPoint.value.y)
  const width = Math.abs(endPoint.value.x - startPoint.value.x)
  const height = Math.abs(endPoint.value.y - startPoint.value.y)
  rect.style.left = x + 'px'
  rect.style.top = y + 'px'
  rect.style.width = width + 'px'
  rect.style.height = height + 'px'
}

const removeSelectionRect = () => {
  if (selectionRect.value) {
    document.body.removeChild(selectionRect.value)
    selectionRect.value = null
  }
}

const selectInRect = () => {
  const x1 = Math.min(startPoint.value.x, endPoint.value.x)
  const y1 = Math.min(startPoint.value.y, endPoint.value.y)
  const x2 = Math.max(startPoint.value.x, endPoint.value.x)
  const y2 = Math.max(startPoint.value.y, endPoint.value.y)

  const cards = document.querySelectorAll('.prelabel-card:not(.confirmed)')
  cards.forEach(card => {
    const rect = card.getBoundingClientRect()
    const cardCenterX = (rect.left + rect.right) / 2
    const cardCenterY = (rect.top + rect.bottom) / 2

    if (cardCenterX >= x1 && cardCenterX <= x2 && cardCenterY >= y1 && cardCenterY <= y2) {
      const checkbox = card.querySelector('.select-checkbox')
      if (checkbox && !checkbox.checked) {
        checkbox.click()
      }
    }
  })
}

const confirmBatch = async () => {
  if (selectedItems.value.length === 0) return
  
  if (!confirm(`确定要批量确认 ${selectedItems.value.length} 个标注吗？`)) {
    return
  }

  const pendingItems = prelabels.value.filter(p => selectedItems.value.includes(p.char_id))
  
  const items = pendingItems.map(item => ({
    char_id: item.char_id,
    char: item.predicted_char
  }))

  try {
    const res = await axios.post('/api/labeling/confirm/batch', { items })
    if (res.data.code === 0) {
      for (const item of pendingItems) {
        const index = prelabels.value.findIndex(p => p.char_id === item.char_id)
        if (index > -1) {
          prelabels.value[index].status = 'confirmed'
        }
      }
      // 重新排序，让已确认的放在最后
      prelabels.value.sort((a, b) => {
        const aConfirmed = a.status === 'confirmed'
        const bConfirmed = b.status === 'confirmed'
        if (aConfirmed && !bConfirmed) return 1
        if (!aConfirmed && bConfirmed) return -1
        const confA = a.confidence || 0
        const confB = b.confidence || 0
        return confA - confB
      })
    }
  } catch (err) {
    console.error('批量确认标注失败:', err)
  }
  
  selectedItems.value = []
  selectAll.value = false
}

const modifyLabel = (index) => {
  modifyItem.value = prelabels.value[index]
  modifyIndex.value = index
  newLabel.value = modifyItem.value.predicted_char
  showModifyModal.value = true
}

const closeModifyModal = () => {
  showModifyModal.value = false
  modifyItem.value = null
  modifyIndex.value = -1
  newLabel.value = ''
}

const saveModifiedLabel = async () => {
  if (!newLabel.value.trim()) return

  try {
    const res = await axios.post('/api/labeling/confirm', {
      char_id: modifyItem.value.char_id,
      char: newLabel.value.trim()
    })
    if (res.data.code === 0) {
      prelabels.value[modifyIndex.value].status = 'confirmed'
      prelabels.value[modifyIndex.value].predicted_char = newLabel.value.trim()
      closeModifyModal()
    }
  } catch (err) {
    console.error('保存标注失败:', err)
  }
}

const skipLabel = (index) => {
  prelabels.value[index].status = 'skipped'
}

const closeLineModal = () => {
  showLineModal.value = false
  lineContextImage.value = ''
  currentLineChar.value = ''
}

const showLineContext = async (item) => {
  // 从lineage获取位置信息
  const lineage = item.lineage || {}

  // 从char_id中提取行名称，例如：page_10_line_5_char_3 -> page_10_line_5
  const match = item.char_id.match(/^(page_\d+_line_\d+)_char_\d+/)
  if (!match) {
    alert('无法获取行信息：char_id格式不正确')
    return
  }

  const lineName = match[1]
  const charIndexInLine = item.char_id.match(/char_(\d+)/) ? parseInt(item.char_id.match(/char_(\d+)/)[1]) : 0

  const lineUrl = `/api/line-images/${encodeURIComponent(lineName)}`

  // 优先使用lineage中的col_start和col_end
  let charLeft = lineage.col_start
  let charRight = lineage.col_end

  try {
    const response = await fetch(lineUrl)

    if (!response.ok) {
      alert(`无法获取行图片：HTTP ${response.status}`)
      return
    }

    const blob = await response.blob()
    const bitmap = await createImageBitmap(blob)

    // 如果没有位置信息，使用估算值
    if (charLeft == null || charRight == null) {
      const avgCharWidth = Math.floor(bitmap.width / 30)
      charLeft = charIndexInLine * avgCharWidth
      charRight = charLeft + avgCharWidth
    }

    const extend = 50
    const cropLeft = Math.max(0, charLeft - extend)
    const cropRight = Math.min(bitmap.width, charRight + extend)
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
    
    const rightLineX = charRight - cropLeft
    ctx.beginPath()
    ctx.moveTo(rightLineX, 0)
    ctx.lineTo(rightLineX, bitmap.height)
    ctx.stroke()

    lineContextImage.value = canvas.toDataURL('image/png')
    currentLineChar.value = item.predicted_char
    showLineModal.value = true
  } catch (err) {
    console.error('加载行图片失败:', err)
    alert(`加载行图片失败：${err.message}`)
  }
}

onMounted(() => {
  loadPreLabels()
  // 添加框选事件监听
  document.addEventListener('mousedown', handleMouseDown)
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
})

onUnmounted(() => {
  // 移除事件监听
  document.removeEventListener('mousedown', handleMouseDown)
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
})
</script>

<style scoped>
.page-container { 
  padding: 20px; 
  max-width: 1400px; 
  margin: 0 auto; 
  min-height: 100vh;
  background: #f5f5f5;
}
.header { 
  display: flex; 
  align-items: center; 
  gap: 16px; 
  margin-bottom: 20px;
  background: #fff;
  padding: 16px 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.title { 
  margin: 0; 
  font-size: 18px;
  font-weight: 600;
}
.header-stats {
  margin-left: auto;
  display: flex;
  gap: 20px;
  font-size: 14px;
  color: #666;
}
.header-stats .highlight {
  color: #1890ff;
  font-weight: bold;
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-left: 20px;
}
.select-all {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  font-size: 14px;
  color: #666;
}
.select-all input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}
.select-all:has(input:disabled) {
  cursor: not-allowed;
  opacity: 0.5;
}
.loading, .empty {
  text-align: center;
  padding: 60px;
  color: #999;
  background: #fff;
  border-radius: 8px;
  margin-top: 20px;
}
.prelabels-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
}
.prelabel-card {
  background: #fff;
  border: 2px solid #d9d9d9;
  border-radius: 8px;
  padding: 8px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}
.select-checkbox {
  position: absolute;
  top: 4px;
  left: 4px;
  width: 16px;
  height: 16px;
  cursor: pointer;
  z-index: 10;
}
.prelabel-card:hover {
  border-color: #1890ff;
}
.prelabel-card.confirmed {
  border-color: #52c41a;
  background: #f6ffed;
}
.prelabel-card.selected {
  border-width: 3px;
  border-style: dashed;
  transform: scale(1.02);
}
.prelabel-card.high-confidence:not(.confirmed) {
  border-color: #d9d9d9;
}
.prelabel-card.medium-confidence:not(.confirmed) {
  border-color: #d9d9d9;
}
.prelabel-card.low-confidence:not(.confirmed) {
  border-color: #d9d9d9;
}
.prelabel-card.high-confidence.selected {
  border-color: #52c41a;
  background: #f6ffed;
  box-shadow: 0 0 12px rgba(82, 196, 26, 0.4);
}
.prelabel-card.medium-confidence.selected {
  border-color: #faad14;
  background: #fffbe6;
  box-shadow: 0 0 12px rgba(250, 173, 20, 0.4);
}
.prelabel-card.low-confidence.selected {
  border-color: #f5222d;
  background: #fff1f0;
  box-shadow: 0 0 12px rgba(245, 34, 45, 0.4);
}
.image-wrapper {
  width: 80px;
  height: 80px;
  margin: 0 auto 8px;
  background: #f5f5f5;
  border-radius: 4px;
  overflow: hidden;
}
.image-wrapper img { 
  width: 100%; 
  height: 100%; 
  object-fit: contain; 
}
.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 4px;
  margin-bottom: 4px;
}
.confidence-badge {
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
  font-size: 10px;
  font-weight: bold;
  padding: 2px 4px;
  border-radius: 4px;
}
.confidence-badge.high {
  background: rgba(82, 196, 26, 0.8);
}
.confidence-badge.medium {
  background: rgba(250, 173, 20, 0.8);
}
.confidence-badge.low {
  background: rgba(245, 34, 45, 0.8);
}
.status-tag {
  font-size: 11px;
  color: #999;
  display: flex;
  align-items: center;
}
.confirmed .status-tag {
  color: #52c41a;
  font-weight: bold;
}
.predicted-char {
  font-size: 20px;
  font-weight: bold;
  color: #333;
  margin-bottom: 8px;
}
.actions {
  display: flex;
  gap: 4px;
  justify-content: center;
}
.actions button {
  font-size: 11px;
  padding: 4px 8px;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal-content {
  background: #fff;
  border-radius: 12px;
  width: 90%;
  max-width: 400px;
  overflow: hidden;
}
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid #e8e8e8;
}
.modal-header h3 {
  margin: 0;
  font-size: 16px;
}
.modal-header button {
  font-size: 24px;
  padding: 0;
  line-height: 1;
}
.modal-body {
  padding: 20px;
}
.modal-image {
  width: 120px;
  height: 120px;
  margin: 0 auto 16px;
  background: #f5f5f5;
  border-radius: 8px;
  overflow: hidden;
}
.modal-image img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.current-label {
  text-align: center;
  margin-bottom: 16px;
  color: #666;
}
.current-label .char {
  font-size: 24px;
  font-weight: bold;
  color: #333;
}
.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 16px 20px;
  border-top: 1px solid #e8e8e8;
}

/* 框选矩形样式 */
.selection-rect {
  position: fixed;
  border: 2px dashed #1890ff;
  background: rgba(24, 144, 255, 0.15);
  pointer-events: none;
  z-index: 9999;
}

/* 行上下文模态框样式 */
.line-modal {
  max-width: 800px;
}

.line-modal-body {
  padding: 16px;
}

.line-modal-hint {
  text-align: center;
  margin-top: 12px;
  font-size: 12px;
  color: #666;
}
</style>