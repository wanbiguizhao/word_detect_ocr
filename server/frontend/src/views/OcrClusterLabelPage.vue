<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack">← 返回列表</a-button>
      <h2 class="title">聚类 {{ clusterId }} 标注</h2>
      <div class="cluster-info">
        <span>别名: </span>
        <a-input v-model:value="alias" placeholder="设置别名" style="width: 120px;" @blur="saveAlias" />
        <span style="margin-left: 16px;">汉字: </span>
        <span class="chars-display">{{ charsDisplay || '?' }}</span>
      </div>
      <a-button type="primary" @click="saveAll" :loading="saving" :disabled="!batchChar && Object.keys(pendingLabels).length === 0">保存全部</a-button>
      <a-button @click="skipCluster" :loading="skipping" danger>暂不标记</a-button>
    </div>

    <div class="selection-toolbar">
      <a-button @click="selectAll">全选</a-button>
      <span>批量标注: </span>
      <a-input v-model:value="batchChar" placeholder="输入汉字" style="width: 80px;" @input="onBatchCharInput" />
      <a-button type="primary" @click="saveAll" :loading="saving" :disabled="!batchChar && selectedCount === 0">
        保存全部 ({{ selectedCount }})
      </a-button>
      <a-button @click="clearSelection">清除选择</a-button>
    </div>

    <div class="recommend-section">
      <div class="recommend-header">
        <span>二次聚类推荐</span>
        <a-select v-model:value="recommendMode" style="width: 140px;" @change="loadRecommendations">
          <a-select-option value="global">全局锚点</a-select-option>
          <a-select-option value="local">当前聚类锚点</a-select-option>
        </a-select>
        <a-button size="small" @click="loadRecommendations">刷新</a-button>
      </div>
      <div class="recommend-tags" v-if="recommendations.length > 0">
        <a-tag
          v-for="rec in recommendations"
          :key="rec.char"
          :color="rec.char === selectedRecommendChar ? 'blue' : 'green'"
          class="recommend-tag"
          @click="selectRecommendChar(rec.char)"
        >
          {{ rec.char }} ({{ rec.count }}张, {{ (rec.avg_similarity * 100).toFixed(0) }}%)
        </a-tag>
      </div>
      <div v-else class="no-recommend">
        <span style="color: #999;">暂无推荐（需要有已标记的标签作为锚点）</span>
      </div>
      <div v-if="selectedRecommendChar && recommendImages.length > 0" class="recommend-images">
        <div class="recommend-images-header">
          <span>推荐 "{{ selectedRecommendChar }}" 的图片（拖拽框选，右键取消推荐）</span>
          <div class="recommend-images-actions">
            <a-button size="small" @click="selectAllRecommendImages">全选</a-button>
            <a-button size="small" @click="clearAllRecommendSelection">清除选择</a-button>
            <a-button type="primary" size="small" @click="applyRecommendLabel">标注选中 ({{ selectedRecommendCount }}张)</a-button>
          </div>
        </div>
        <div
          class="recommend-images-grid"
          ref="recommendContainerRef"
          @mousedown="startRecommendSelection"
          @mousemove="updateRecommendSelection"
          @mouseup="endRecommendSelection"
          @mouseleave="endRecommendSelection"
        >
          <div
            v-for="img in sortedRecommendImages"
            :key="img.index"
            :data-index="img.index"
            class="image-card recommend-image-card"
            :class="{ selected: selectedRecommendIndices.includes(img.index) }"
            @click="toggleRecommendSelect(img.index)"
            @contextmenu.prevent="handleRecommendContextMenu($event, img.index)"
          >
            <div class="image-wrapper">
              <img :src="getImageUrl(img.char_id)" :alt="img.char_id" />
            </div>
            <div class="similarity-badge">{{ (img.similarity * 100).toFixed(0) }}%</div>
          </div>
          <div
            v-if="isRecommendSelecting"
            class="selection-box"
            :style="recommendSelectionBoxStyle"
          ></div>
        </div>
      </div>
    </div>

    <div
      class="images-container"
      ref="containerRef"
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
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'

const route = useRoute()
const router = useRouter()

const clusterId = computed(() => route.params.id)
const images = ref([])
const pendingLabels = ref({})
const alias = ref('')
const batchChar = ref('')
const charsDisplay = ref('')
const saving = ref(false)
const skipping = ref(false)
const showLineModal = ref(false)
const lineContextImage = ref('')
const contextMenuIndex = ref(null)

const recommendMode = ref('global')
const recommendations = ref([])
const selectedRecommendChar = ref('')
const recommendImages = ref([])
const selectedRecommendIndices = ref([])
const recommendContainerRef = ref(null)
const isRecommendSelecting = ref(false)
const recommendSelectionStart = ref({ x: 0, y: 0 })
const recommendSelectionEnd = ref({ x: 0, y: 0 })

// 添加调试日志，追踪 recommendMode 变化
import { watch } from 'vue'
watch(recommendMode, (newVal, oldVal) => {
})

const selectedRecommendCount = computed(() => selectedRecommendIndices.value.length)

// 排序推荐图片：选中的在前，未选中的在后
const sortedRecommendImages = computed(() => {
  return [...recommendImages.value].sort((a, b) => {
    const aSelected = selectedRecommendIndices.value.includes(a.index)
    const bSelected = selectedRecommendIndices.value.includes(b.index)
    if (aSelected && !bSelected) return -1
    if (!aSelected && bSelected) return 1
    // 未选中的按相似度排序
    return b.similarity - a.similarity
  })
})

const onBatchCharInput = () => {
}

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
  return `/api/char-images/${charId}.png`
}

const showLineContext = async (img) => {
  if (!img.lineage) {
    alert('无法获取行信息：缺少血缘数据')
    return
  }

  if (!img.lineage.line_name) {
    alert('无法获取行信息：缺少行名称')
    return
  }

  const lineName = img.lineage.line_name
  const lineUrl = `/api/line-images/${encodeURIComponent(lineName)}`

  let charLeft = img.lineage.col_start
  let charRight = undefined
  
  if (charLeft !== undefined && img.lineage.col_end !== undefined) {
    charRight = img.lineage.col_end
  } else if (charLeft !== undefined && img.lineage.width !== undefined) {
    charRight = charLeft + img.lineage.width
  }

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
    showLineModal.value = true
  } catch (err) {
    console.error('加载行图片失败:', err)
    alert(`加载行图片失败：${err.message}`)
  }
}

const loadData = async () => {
  try {
    const res = await axios.get(`/api/clusters/${clusterId.value}/images`)
    images.value = res.data.images || []

    const labelsRes = await axios.get('/api/cluster-labels')
    const clusterLabel = labelsRes.data.data?.[clusterId.value] || {}
    alias.value = clusterLabel.alias || ''

    const charsMap = clusterLabel.chars || {}
    charsDisplay.value = Object.entries(charsMap).length > 0
      ? Object.entries(charsMap)
          .sort((a, b) => b[1] - a[1])
          .map(([char, count]) => `${char}(${count})`)
          .join(', ')
      : (clusterLabel.char ? `${clusterLabel.char}(${images.value.length})` : '')

    // 如果聚类已有标注，优先使用当前聚类锚点
    const hasLabels = clusterLabel.char_labels && Object.keys(clusterLabel.char_labels).length > 0
    if (hasLabels && recommendMode.value === 'global') {
      recommendMode.value = 'local'
    }
  } catch (err) {
    console.error('加载数据失败:', err)
  }
}

const loadRecommendations = async () => {
  try {
    const res = await axios.get(`/api/clusters/${clusterId.value}/recommend?mode=${recommendMode.value}`)
    if (res.data.code === 0 && res.data.recommendations) {
      recommendations.value = Object.entries(res.data.recommendations).map(([char, info]) => ({
        char,
        count: info.count,
        avg_similarity: info.avg_similarity
      }))
      selectedRecommendChar.value = ''
      recommendImages.value = []
      selectedRecommendIndices.value = []
    }
  } catch (err) {
    console.error('加载推荐失败:', err)
  }
}

const selectRecommendChar = async (char) => {
  selectedRecommendChar.value = char
  selectedRecommendIndices.value = []
  try {
    const res = await axios.get(`/api/clusters/${clusterId.value}/recommend/${char}`)
    if (res.data.code === 0) {
      recommendImages.value = res.data.images || []
    }
  } catch (err) {
    console.error('加载推荐图片失败:', err)
  }
}

const toggleRecommendSelect = (index) => {
  const idx = selectedRecommendIndices.value.indexOf(index)
  if (idx === -1) {
    selectedRecommendIndices.value.push(index)
  } else {
    selectedRecommendIndices.value.splice(idx, 1)
  }
}

const handleRecommendContextMenu = (event, index) => {
  event.preventDefault()
  if (confirm('确定要从推荐中移除这张图片吗？')) {
    recommendImages.value = recommendImages.value.filter(img => img.index !== index)
    const selectedIdx = selectedRecommendIndices.value.indexOf(index)
    if (selectedIdx !== -1) {
      selectedRecommendIndices.value.splice(selectedIdx, 1)
    }
  }
}

const selectAllRecommendImages = () => {
  selectedRecommendIndices.value = recommendImages.value.map(img => img.index)
}

const clearAllRecommendSelection = () => {
  selectedRecommendIndices.value = []
}

const startRecommendSelection = (event) => {
  if (event.target.closest('.recommend-image-card')) return

  isRecommendSelecting.value = true
  const rect = recommendContainerRef.value.getBoundingClientRect()
  recommendSelectionStart.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  }
  recommendSelectionEnd.value = { ...recommendSelectionStart.value }
}

const updateRecommendSelection = (event) => {
  if (!isRecommendSelecting.value) return

  const rect = recommendContainerRef.value.getBoundingClientRect()
  recommendSelectionEnd.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  }
}

const endRecommendSelection = (event) => {
  if (!isRecommendSelecting.value) return

  isRecommendSelecting.value = false

  const rect = recommendContainerRef.value.getBoundingClientRect()
  const boxLeft = Math.min(recommendSelectionStart.value.x, recommendSelectionEnd.value.x)
  const boxRight = Math.max(recommendSelectionStart.value.x, recommendSelectionEnd.value.x)
  const boxTop = Math.min(recommendSelectionStart.value.y, recommendSelectionEnd.value.y)
  const boxBottom = Math.max(recommendSelectionStart.value.y, recommendSelectionEnd.value.y)

  const cardElements = recommendContainerRef.value.querySelectorAll('.recommend-image-card')
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
      if (!selectedRecommendIndices.value.includes(index)) {
        newSelected.push(index)
      }
    }
  })

  selectedRecommendIndices.value = [...selectedRecommendIndices.value, ...newSelected]
}

const recommendSelectionBoxStyle = computed(() => {
  const left = Math.min(recommendSelectionStart.value.x, recommendSelectionEnd.value.x)
  const top = Math.min(recommendSelectionStart.value.y, recommendSelectionEnd.value.y)
  const width = Math.abs(recommendSelectionEnd.value.x - recommendSelectionStart.value.x)
  const height = Math.abs(recommendSelectionEnd.value.y - recommendSelectionStart.value.y)
  return {
    left: left + 'px',
    top: top + 'px',
    width: width + 'px',
    height: height + 'px'
  }
})

const applyRecommendLabel = async () => {
  if (!selectedRecommendChar.value || selectedRecommendIndices.value.length === 0) return

  saving.value = true
  try {
    for (const idx of selectedRecommendIndices.value) {
      pendingLabels.value[idx] = selectedRecommendChar.value
    }

    const response = await axios.post('/api/cluster-labels/batch-save', {
      clusterId: parseInt(clusterId.value),
      labels: selectedRecommendIndices.value.map(idx => ({
        char: selectedRecommendChar.value,
        charIndex: idx
      }))
    })

    if (response.data.new_chars_count > 0) {
      const newChars = response.data.new_chars.join('、')
      alert(`标注成功！\n\n新增汉字 ${response.data.new_chars_count} 个：\n${newChars}`)
    } else {
      alert('标注成功！\n\n暂无新增汉字')
    }
    window.location.reload()
  } catch (err) {
    console.error('标注失败:', err)
    alert('标注失败')
  } finally {
    saving.value = false
  }
}

const skipCluster = async () => {
  if (!confirm('确定要标记该聚类为"暂不标记"吗？标记后可以在列表中过滤查看。')) {
    return
  }

  skipping.value = true
  try {
    await axios.post(`/api/clusters/${clusterId.value}/skip`)
    alert('已标记为暂不标记')
    router.push('/ocr')
  } catch (err) {
    console.error('标记失败:', err)
    alert('标记失败')
  } finally {
    skipping.value = false
  }
}

const updatePendingLabel = (index, event) => {
  const value = event.target.value
  if (value) {
    pendingLabels.value[index] = value
  } else {
    delete pendingLabels.value[index]
  }
}

const saveLabel = async (img) => {
  // 保持空实现，让标注停留在 pendingLabels 中
  // 只有点击"保存全部"才真正保存
}

const saveAlias = async () => {
  try {
    await axios.post('/api/cluster-labels/save', {
      clusterId: parseInt(clusterId.value),
      alias: alias.value
    })
  } catch (err) {
    console.error('保存别名失败:', err)
  }
}

const batchLabelSelected = async () => {
  if (!batchChar.value || selectedIndices.value.length === 0) {
    return
  }


  for (const idx of selectedIndices.value) {
    const img = images.value.find(i => i.index === idx)
    if (img && !img.label) {
      pendingLabels.value[idx] = batchChar.value
    }
  }

  selectedIndices.value = []
  batchChar.value = ''
}

const selectAll = () => {
  selectedIndices.value = images.value.map(img => img.index)
}

const clearSelection = () => {
  selectedIndices.value = []
}

const toggleSelect = (index, event) => {
  const idx = selectedIndices.value.indexOf(index)
  if (idx === -1) {
    selectedIndices.value.push(index)
  } else {
    selectedIndices.value.splice(idx, 1)
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
      if (!selectedIndices.value.includes(index)) {
        newSelected.push(index)
      }
    }
  })

  selectedIndices.value = [...selectedIndices.value, ...newSelected]
}

const saveAll = async () => {
  saving.value = true


  try {
    let toSave = []

    if (batchChar.value && selectedIndices.value.length > 0) {
      for (const idx of selectedIndices.value) {
        toSave.push({ index: idx, char: batchChar.value })
      }
    }

    for (const [index, char] of Object.entries(pendingLabels.value)) {
      toSave.push({ index: parseInt(index), char: char })
    }


    if (toSave.length === 0) {
      saving.value = false
      return
    }

    const response = await axios.post('/api/cluster-labels/batch-save', {
      clusterId: parseInt(clusterId.value),
      labels: toSave.map(item => ({ char: item.char, charIndex: item.index }))
    })

    pendingLabels.value = {}
    selectedIndices.value = []
    batchChar.value = ''
    
    if (response.data.new_chars_count > 0) {
      const newChars = response.data.new_chars.join('、')
      alert(`保存成功！\n\n新增汉字 ${response.data.new_chars_count} 个：\n${newChars}`)
    } else {
      alert('保存成功！\n\n暂无新增汉字')
    }
    
    window.location.reload()
  } catch (err) {
    console.error('保存失败:', err)
    alert('保存失败')
  } finally {
    saving.value = false
  }
}

const goBack = () => {
  router.push('/ocr')
}

onMounted(() => {
  loadData().then(() => {
    loadRecommendations()
  })
})
</script>

<style scoped>
.page-container { padding: 20px; max-width: 1400px; margin: 0 auto; }
.header { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
.title { margin: 0; }
.cluster-info { display: flex; align-items: center; margin-right: auto; }
.suggested-char { font-size: 24px; font-weight: bold; color: #1890ff; }
.chars-display { font-size: 18px; font-weight: bold; color: #1890ff; }
.selection-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  padding: 12px;
  background: #f5f5f5;
  border-radius: 8px;
}
.images-container {
  position: relative;
  min-height: 400px;
  border: 1px dashed #ccc;
  border-radius: 8px;
  padding: 16px;
  background: #fafafa;
  user-select: none;
}
.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
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
  box-shadow: 0 0 12px rgba(24, 144, 255, 0.4);
  transform: scale(1.02);
}
.image-card.labeled { border-color: #52c41a; background: #f6ffed; }
.image-card.pending { 
  border-color: #faad14; 
  background: #fffbe6; 
  border-style: dashed;
  animation: pending-pulse 1.5s ease-in-out infinite;
}
@keyframes pending-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(250, 173, 20, 0.4); }
  50% { box-shadow: 0 0 0 6px rgba(250, 173, 20, 0); }
}
.image-wrapper { width: 80px; height: 80px; margin: 0 auto 8px; }
.image-wrapper img { width: 100%; height: 100%; object-fit: contain; }
.label-area { margin-bottom: 4px; }
.char-id { font-size: 10px; color: #999; word-break: break-all; }
.selection-box {
  position: absolute;
  border: 2px dashed #1890ff;
  background: rgba(24, 144, 255, 0.1);
  pointer-events: none;
  z-index: 100;
}
.stats-bar {
  display: flex;
  gap: 24px;
  margin-top: 20px;
  padding: 12px;
  background: #f5f5f5;
  border-radius: 8px;
}
.recommend-section {
  margin-bottom: 20px;
  padding: 16px;
  background: #fffbe6;
  border: 1px solid #ffe58f;
  border-radius: 8px;
}
.recommend-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  font-weight: 500;
}
.recommend-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.recommend-tag {
  cursor: pointer;
  padding: 4px 12px;
  font-size: 14px;
}
.recommend-images {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px dashed #ffe58f;
}
.recommend-images-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.recommend-images-actions {
  display: flex;
  gap: 8px;
}
.recommend-images-grid {
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
.recommend-image-card {
  position: relative;
  background: #e6f7ff;
  border: 2px solid #91d5ff;
}
.recommend-image-card.selected {
  border-color: #1890ff;
  background: #bae7ff;
}
.similarity-badge {
  position: absolute;
  bottom: 4px;
  right: 4px;
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
  font-size: 10px;
  padding: 2px 4px;
  border-radius: 4px;
}
</style>
