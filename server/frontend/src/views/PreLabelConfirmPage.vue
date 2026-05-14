<template>
  <div class="page-container">
    <div class="header">
      <!-- 第一行：标题和统计 -->
      <div class="header-section">
        <a-button class="back-btn" @click="goBack">← 返回</a-button>
        <h1 class="title">预标注详情 - <span class="char-highlight">{{ char }}</span></h1>
        <div class="header-stats">
          <div class="stat-item">
            <span class="stat-label">总数</span>
            <span class="stat-value">{{ total }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">已确认</span>
            <span class="stat-value confirmed">{{ confirmed }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">待确认</span>
            <span class="stat-value pending">{{ pending }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">已跳过</span>
            <span class="stat-value skipped">{{ skipped }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">已选</span>
            <span class="stat-value selected">{{ selectedCount }}</span>
          </div>
        </div>
      </div>
      
      <!-- 第二行：置信度按钮 -->
      <div class="header-section">
        <div class="action-group">
          <span class="group-label">置信度</span>
          <a-button @click="selectByConfidence('high')">
            🟢 选择高置信度
          </a-button>
          <a-button @click="selectByConfidence('medium')">
            🟠 选择中置信度
          </a-button>
          <a-button @click="selectByConfidence('low')">
            🔴 选择低置信度
          </a-button>
        </div>
      </div>
      
      <!-- 第三行：操作按钮 -->
      <div class="header-section actions-section">
        <div class="action-group">
          <span class="group-label">选择</span>
          <a-button 
            @click="toggleSelectAll"
            :disabled="prelabels.length === 0"
          >
            {{ isAllSelected ? '✓ 全选' : '☐ 全选' }}
          </a-button>
          <a-button 
            :disabled="selectedItems.length === 0"
            @click="clearSelection"
          >
            取消
          </a-button>
        </div>
        
        <div class="action-group">
          <span class="group-label">批量操作</span>
          <a-button 
            type="primary" 
            :disabled="selectedItems.length === 0"
            @click="confirmBatch"
          >
            ✓ 确认 ({{ selectedItems.length }})
          </a-button>
          <a-button 
            :disabled="selectedItems.length === 0"
            @click="skipBatch"
          >
            ⏭ 跳过 ({{ selectedItems.length }})
          </a-button>
          <a-button 
            :disabled="selectedItems.length === 0"
            @click="revokeBatch"
          >
            ↶ 撤回 ({{ selectedItems.length }})
          </a-button>
          <a-button 
            :disabled="selectedItems.length === 0"
            @click="unskipBatch"
          >
            ↶ 取消跳过 ({{ selectedItems.length }})
          </a-button>
        </div>
        <div class="action-group">
          <span class="group-label">批量修改</span>
          <div class="batch-modify-group">
            <input 
              v-model="batchModifyChar"
              placeholder="输入汉字"
              style="width: 100px; padding: 4px 8px; border: 1px solid #d9d9d9; border-radius: 4px;"
              @keyup.enter="modifyBatch"
            />
            <a-button 
              type="primary"
              :disabled="isModifyDisabled"
              @click="modifyBatch"
            >
              修改
            </a-button>
          </div>
        </div>
      </div>
    </div>

    <div v-if="loading" class="loading">加载中...</div>

    <div v-else-if="prelabels.length === 0" class="empty">
      暂无预标注数据
    </div>

    <div 
      v-else 
      class="prelabels-container"
      ref="containerRef"
      @mousedown="startSelection"
      @mousemove="updateSelection"
      @mouseup="endSelection"
      @mouseleave="endSelection"
    >
      <div class="prelabels-grid">
        <div
          v-for="(item, index) in sortedPrelabels"
          :key="item.char_id"
          :data-char-id="item.char_id"
          class="prelabel-card"
          :class="{ 
            'high-confidence': item.confidence_level === 'high',
            'medium-confidence': item.confidence_level === 'medium',
            'low-confidence': item.confidence_level === 'low',
            'confirmed': item.status === 'confirmed',
            'skipped': item.status === 'skipped',
            'selected': isSelected(item.char_id),
            'modified': !!item.corrected_char
          }"
          @click="toggleSelect(item.char_id)"
          @contextmenu.prevent="showLineContext(item)"
        >
          <div class="image-wrapper">
            <img :src="getImageUrl(item.image_path)" :alt="item.char_id" />
          </div>
          <div class="info-row">
            <div class="status-tag" :class="item.status">
              {{ item.status === 'confirmed' ? '已确认' : item.status === 'skipped' ? '已跳过' : '待确认' }}
            </div>
            <div class="confidence-badge" :class="item.confidence_level">
              {{ (item.confidence * 100).toFixed(0) }}%
            </div>
          </div>
          <div class="predicted-char" :class="getPredictedCharClass(item)">
            {{ item.corrected_char || item.predicted_char }}
          </div>
        </div>
      </div>
      
      <!-- 选择框 + 实时计数 -->
      <div
        v-if="isSelecting && hasMoved"
        class="selection-box"
        :style="selectionBoxStyle"
      >
        <span class="selection-count">{{ liveSelectionCount }}</span>
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
import { message } from 'ant-design-vue'

const route = useRoute()
const router = useRouter()
const char = decodeURIComponent(route.params.char)

const loading = ref(false)
const prelabels = ref([])
const selectedItems = ref([])
const batchModifyChar = ref('')

// 框选相关状态
const containerRef = ref(null)
const isSelecting = ref(false)
const hasMoved = ref(false)
const dragJustEnded = ref(false)
const liveSelectionCount = ref(0)
const selectionStart = ref({ x: 0, y: 0 })
const selectionEnd = ref({ x: 0, y: 0 })

// 行上下文相关状态
const showLineModal = ref(false)
const lineContextImage = ref('')
const currentLineChar = ref('')

// 计算属性
const total = computed(() => prelabels.value.length)
const confirmed = computed(() => prelabels.value.filter(p => p.status === 'confirmed').length)
const pending = computed(() => prelabels.value.filter(p => p.status === 'pending').length)
const skipped = computed(() => prelabels.value.filter(p => p.status === 'skipped').length)
const selectedCount = computed(() => selectedItems.value.length)

const isAllSelected = computed(() => {
  return prelabels.value.length > 0 && 
    prelabels.value.every(p => selectedItems.value.includes(p.char_id))
})

const isModifyDisabled = computed(() => {
  return selectedItems.value.length === 0 || !batchModifyChar.value?.trim()
})

// 按分组排序但不拆分显示
const sortedPrelabels = computed(() => {
  const groups = {}
  
  // 按显示字符分组
  prelabels.value.forEach(item => {
    const displayChar = item.corrected_char || item.predicted_char
    if (!groups[displayChar]) {
      groups[displayChar] = []
    }
    groups[displayChar].push(item)
  })
  
  // 转换为数组并按数量升序排序
  const sortedGroups = Object.entries(groups)
    .sort(([,a], [,b]) => a.length - b.length)
  
  // 展开并添加分组索引，同时计算显示优先级
  const result = []
  sortedGroups.forEach(([char, items], groupIndex) => {
    // 组内按置信度由低到高排序
    const sortedItems = items.sort((a, b) => a.confidence - b.confidence)
    // 给每个项目添加分组索引和显示优先级
    sortedItems.forEach(item => {
      // 显示优先级：0=待确认，1=预测错误（已修正），2=预测正确
      let sortPriority
      if (item.status === 'pending') {
        sortPriority = 0
      } else if (item.corrected_char) {
        sortPriority = 1
      } else {
        sortPriority = 2
      }
      result.push({
        ...item,
        groupIndex,
        sortPriority
      })
    })
  })
  
  // 按优先级排序：先按显示优先级，再按分组索引，最后按置信度
  result.sort((a, b) => {
    if (a.sortPriority !== b.sortPriority) {
      return a.sortPriority - b.sortPriority
    }
    if (a.groupIndex !== b.groupIndex) {
      return a.groupIndex - b.groupIndex
    }
    return a.confidence - b.confidence
  })
  
  return result
})

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

// 方法
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
      prelabels.value = res.data.data || []
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

const getPredictedCharClass = (item) => {
  const classes = []
  // 只有修改过的字才显示分组颜色
  if (item.corrected_char) {
    if (item.groupIndex !== undefined) {
      if (item.groupIndex % 2 === 0) {
        classes.push('group-red')
      } else {
        classes.push('group-yellow')
      }
    }
  }
  // 保持modified-char的标识
  if (item.corrected_char) {
    classes.push('modified-char')
  }
  return classes.join(' ')
}

const toggleSelect = (charId) => {
  if (dragJustEnded.value) return
  const index = selectedItems.value.indexOf(charId)
  if (index > -1) {
    selectedItems.value.splice(index, 1)
  } else {
    selectedItems.value.push(charId)
  }
}

const toggleSelectAll = () => {
  if (isAllSelected.value) {
    selectedItems.value = []
  } else {
    selectedItems.value = prelabels.value.map(p => p.char_id)
  }
}

const clearSelection = () => {
  selectedItems.value = []
}

const selectByConfidence = (level) => {
  const itemsToSelect = prelabels.value.filter(p => p.confidence_level === level)
  itemsToSelect.forEach(item => {
    if (!selectedItems.value.includes(item.char_id)) {
      selectedItems.value.push(item.char_id)
    }
  })
}



// 框选方法
const DRAG_THRESHOLD = 5

const startSelection = (event) => {
  hasMoved.value = false
  liveSelectionCount.value = 0
  isSelecting.value = true
  const rect = containerRef.value.getBoundingClientRect()
  selectionStart.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  }
  selectionEnd.value = { ...selectionStart.value }
}

const computeSelectionHits = (boxLeft, boxRight, boxTop, boxBottom) => {
  const rect = containerRef.value.getBoundingClientRect()
  const cardElements = containerRef.value.querySelectorAll('.prelabel-card')
  const hits = []
  cardElements.forEach((card) => {
    const cardRect = card.getBoundingClientRect()
    const cardLeft = cardRect.left - rect.left
    const cardRight = cardRect.right - rect.left
    const cardTop = cardRect.top - rect.top
    const cardBottom = cardRect.bottom - rect.top
    if (cardRight > boxLeft && cardLeft < boxRight &&
        cardBottom > boxTop && cardTop < boxBottom) {
      const charId = card.getAttribute('data-char-id')
      hits.push(charId)
    }
  })
  return hits
}

const updateSelection = (event) => {
  if (!isSelecting.value) return

  const rect = containerRef.value.getBoundingClientRect()
  selectionEnd.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  }
  const dx = selectionEnd.value.x - selectionStart.value.x
  const dy = selectionEnd.value.y - selectionStart.value.y
  if (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD) {
    hasMoved.value = true
  }

  if (hasMoved.value) {
    const boxLeft = Math.min(selectionStart.value.x, selectionEnd.value.x)
    const boxRight = Math.max(selectionStart.value.x, selectionEnd.value.x)
    const boxTop = Math.min(selectionStart.value.y, selectionEnd.value.y)
    const boxBottom = Math.max(selectionStart.value.y, selectionEnd.value.y)
    liveSelectionCount.value = computeSelectionHits(boxLeft, boxRight, boxTop, boxBottom).length
  }
}

const endSelection = () => {
  if (!isSelecting.value) return
  isSelecting.value = false

  if (hasMoved.value) {
    dragJustEnded.value = true
    setTimeout(() => { dragJustEnded.value = false }, 0)
    const boxLeft = Math.min(selectionStart.value.x, selectionEnd.value.x)
    const boxRight = Math.max(selectionStart.value.x, selectionEnd.value.x)
    const boxTop = Math.min(selectionStart.value.y, selectionEnd.value.y)
    const boxBottom = Math.max(selectionStart.value.y, selectionEnd.value.y)
    const hits = computeSelectionHits(boxLeft, boxRight, boxTop, boxBottom)
    const newItems = hits.filter(id => !selectedItems.value.includes(id))
    if (newItems.length > 0) {
      selectedItems.value = [...selectedItems.value, ...newItems]
    }
  }
  hasMoved.value = false
  liveSelectionCount.value = 0
}

// 批量操作
const confirmBatch = async () => {
  if (selectedItems.value.length === 0) return
  
  const alreadyConfirmed = prelabels.value.filter(
    p => selectedItems.value.includes(p.char_id) && p.status === 'confirmed'
  ).length
  const itemsToConfirm = prelabels.value.filter(
    p => selectedItems.value.includes(p.char_id) && p.status !== 'confirmed'
  )
  
  if (itemsToConfirm.length === 0) {
    message.warning('所选标注已全部确认，无需重复操作')
    return
  }

  let confirmMsg = `确定要批量确认 ${itemsToConfirm.length} 个标注吗？`
  if (alreadyConfirmed > 0) {
    confirmMsg = `已排除 ${alreadyConfirmed} 个已确认项，将确认剩余 ${itemsToConfirm.length} 个标注，确定吗？`
  }
  if (!confirm(confirmMsg)) return

  const itemsData = itemsToConfirm.map(item => ({
    char_id: item.char_id,
    char: item.corrected_char || item.predicted_char
  }))

  try {
    const res = await axios.post('/api/labeling/confirm/batch', { items: itemsData })
    if (res.data.code === 0) {
      const successCount = res.data.success_count ?? 0
      if (successCount > 0) {
        itemsToConfirm.forEach(item => {
          const idx = prelabels.value.findIndex(p => p.char_id === item.char_id)
          if (idx > -1) {
            prelabels.value[idx].status = 'confirmed'
          }
        })
        message.success(`✓ 批量确认成功！已确认 ${successCount} 个标注`)
      } else {
        message.warning('批量确认失败，无标注被确认')
      }
    } else {
      message.error('批量确认失败: ' + (res.data.msg || '未知错误'))
    }
  } catch (err) {
    console.error('批量确认标注失败:', err)
    message.error('批量确认失败，请稍后重试')
  }
  
  selectedItems.value = []
}

const skipBatch = async () => {
  if (selectedItems.value.length === 0) return
  
  const alreadySkipped = prelabels.value.filter(
    p => selectedItems.value.includes(p.char_id) && p.status === 'skipped'
  ).length
  const itemsToSkip = selectedItems.value.filter(charId => {
    const p = prelabels.value.find(x => x.char_id === charId)
    return p && p.status !== 'skipped'
  })

  if (itemsToSkip.length === 0) {
    message.warning('所选标注已全部跳过，无需重复操作')
    return
  }

  let confirmMsg = `确定要批量跳过 ${itemsToSkip.length} 个标注吗？`
  if (alreadySkipped > 0) {
    confirmMsg = `已排除 ${alreadySkipped} 个已跳过项，将跳过剩余 ${itemsToSkip.length} 个标注，确定吗？`
  }
  if (!confirm(confirmMsg)) return

  try {
    const res = await axios.post('/api/labeling/skip/batch', {
      char_ids: itemsToSkip
    })
    if (res.data.code === 0) {
      itemsToSkip.forEach(charId => {
        const idx = prelabels.value.findIndex(p => p.char_id === charId)
        if (idx > -1) {
          prelabels.value[idx].status = 'skipped'
        }
      })
      message.success(`✓ 批量跳过成功！已跳过 ${itemsToSkip.length} 个标注`)
    } else {
      message.error('批量跳过失败: ' + (res.data.msg || '未知错误'))
    }
  } catch (err) {
    console.error('批量跳过失败:', err)
    message.error('批量跳过失败，请稍后重试')
  }
  
  selectedItems.value = []
}

const revokeBatch = async () => {
  if (selectedItems.value.length === 0) return
  
  const itemsToRevoke = prelabels.value.filter(
    p => selectedItems.value.includes(p.char_id) && (p.status === 'confirmed' || p.status === 'skipped')
  )

  if (itemsToRevoke.length === 0) {
    message.warning('所选标注中没有已确认或已跳过的项')
    return
  }

  const alreadyPending = selectedItems.value.length - itemsToRevoke.length
  let confirmMsg = `确定要撤回 ${itemsToRevoke.length} 个标注吗？`
  if (alreadyPending > 0) {
    confirmMsg = `已排除 ${alreadyPending} 个待确认项，将撤回剩余 ${itemsToRevoke.length} 个标注，确定吗？`
  }
  if (!confirm(confirmMsg)) return

  try {
    const res = await axios.post('/api/labeling/revoke/batch', {
      char_ids: itemsToRevoke.map(p => p.char_id)
    })
    if (res.data.code === 0) {
      itemsToRevoke.forEach(item => {
        const idx = prelabels.value.findIndex(p => p.char_id === item.char_id)
        if (idx > -1) {
          prelabels.value[idx].status = 'pending'
        }
      })
      const successCount = res.data.success_count || itemsToRevoke.length
      message.success(`✓ 批量撤回成功！已撤回 ${successCount} 个标注`)
    } else {
      message.error('批量撤回失败: ' + (res.data.msg || '未知错误'))
    }
  } catch (err) {
    console.error('批量撤回失败:', err)
    message.error('批量撤回失败，请稍后重试')
  }
  
  selectedItems.value = []
}

const unskipBatch = async () => {
  if (selectedItems.value.length === 0) return
  
  const itemsToUnskip = prelabels.value.filter(
    p => selectedItems.value.includes(p.char_id) && p.status === 'skipped'
  )

  if (itemsToUnskip.length === 0) {
    message.warning('选中的标注中没有已跳过的标注')
    return
  }

  const notSkipped = selectedItems.value.length - itemsToUnskip.length
  let confirmMsg = `确定要取消跳过 ${itemsToUnskip.length} 个标注吗？`
  if (notSkipped > 0) {
    confirmMsg = `已排除 ${notSkipped} 个非跳过项，将取消跳过 ${itemsToUnskip.length} 个标注，确定吗？`
  }
  if (!confirm(confirmMsg)) return

  try {
    const res = await axios.post('/api/labeling/unskip/batch', {
      char_ids: itemsToUnskip.map(p => p.char_id)
    })
    if (res.data.code === 0) {
      itemsToUnskip.forEach(item => {
        const idx = prelabels.value.findIndex(p => p.char_id === item.char_id)
        if (idx > -1) {
          prelabels.value[idx].status = 'pending'
        }
      })
      const successCount = res.data.success_count || itemsToUnskip.length
      message.success(`✓ 批量取消跳过成功！已取消跳过 ${successCount} 个标注`)
    } else {
      message.error('批量取消跳过失败: ' + (res.data.msg || '未知错误'))
    }
  } catch (err) {
    console.error('批量取消跳过失败:', err)
    message.error('批量取消跳过失败，请稍后重试')
  }
  
  selectedItems.value = []
}

const modifyBatch = async () => {
  if (selectedItems.value.length === 0 || !batchModifyChar.value?.trim()) {
    return
  }
  
  if (!confirm(`确定要将选中的 ${selectedItems.value.length} 个标注修改为"${batchModifyChar.value}"吗？`)) {
    return
  }

  try {
    const res = await axios.post('/api/labeling/modify/batch', {
      char_ids: selectedItems.value,
      new_char: batchModifyChar.value.trim()
    })
    
    if (res.data.code === 0) {
      const updatedPrelabels = [...prelabels.value]
      
      for (const charId of selectedItems.value) {
        const index = updatedPrelabels.findIndex(p => p.char_id === charId)
        if (index !== -1) {
          updatedPrelabels[index] = {
            ...updatedPrelabels[index],
            corrected_char: batchModifyChar.value.trim(),
            status: 'confirmed'
          }
        }
      }
      
      prelabels.value = updatedPrelabels
      
      const successCount = res.data.success_count || selectedItems.value.length
      message.success(`✓ 批量修改成功！已修改 ${successCount} 个标注`)
      batchModifyChar.value = ''
    } else {
      message.error('批量修改失败: ' + (res.data.msg || '未知错误'))
    }
  } catch (err) {
    console.error('批量修改错误:', err)
    message.error('批量修改失败，请稍后重试')
  }
  
  selectedItems.value = []
}

// 行上下文
const showLineContext = async (item) => {
  try {
    const match = item.char_id.match(/^(page_\d+_line_\d+)_char_\d+/)
    if (!match) {
      alert('无法获取行信息：char_id格式不正确')
      return
    }

    const lineName = match[1]
    const lineUrl = `/api/line-images/${encodeURIComponent(lineName)}`

    let charLeft = 0
    let charRight = 100
    
    // 先尝试获取位置信息
    if (item.lineage) {
      charLeft = item.lineage.col_start || 0
      if (item.lineage.col_end !== undefined) {
        charRight = item.lineage.col_end
      } else if (item.lineage.width !== undefined) {
        charRight = charLeft + item.lineage.width
      }
    }

    // 如果没有位置信息，从char_id估算
    if (charLeft === 0 && charRight === 100) {
      const charIndexMatch = item.char_id.match(/_char_(\d+)/)
      const charIndex = charIndexMatch ? parseInt(charIndexMatch[1]) : 0
      charLeft = charIndex * 60
      charRight = charLeft + 60
    }

    const response = await fetch(lineUrl)
    const blob = await response.blob()
    const bitmap = await createImageBitmap(blob)

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
    currentLineChar.value = item.corrected_char || item.predicted_char
    showLineModal.value = true
  } catch (err) {
    console.error('加载行图片失败:', err)
    alert('加载行图片失败')
  }
}

const closeLineModal = () => {
  showLineModal.value = false
  lineContextImage.value = ''
  currentLineChar.value = ''
}

onMounted(() => {
  loadPreLabels()
})

onUnmounted(() => {
  if (lineContextImage.value) {
    URL.revokeObjectURL(lineContextImage.value)
  }
})
</script>

<style scoped>
.page-container {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.header {
  background: white;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.header-section {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
}

.header-section:last-child {
  margin-bottom: 0;
}

.back-btn {
  margin-right: 16px;
}

.title {
  margin: 0;
  font-size: 24px;
  color: #333;
}

.char-highlight {
  color: #1890ff;
  font-weight: bold;
}

.header-stats {
  display: flex;
  gap: 24px;
  margin-left: auto;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-label {
  font-size: 12px;
  color: #999;
  margin-bottom: 4px;
}

.stat-value {
  font-size: 18px;
  font-weight: bold;
  color: #333;
}

.stat-value.confirmed {
  color: #52c41a;
}

.stat-value.pending {
  color: #faad14;
}

.stat-value.skipped {
  color: #999;
}

.stat-value.selected {
  color: #1890ff;
}

.action-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.group-label {
  font-size: 14px;
  color: #666;
  margin-right: 8px;
}

.actions-section {
  flex-wrap: wrap;
  gap: 24px;
}

.batch-modify-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.loading,
.empty {
  text-align: center;
  padding: 60px 20px;
  color: #999;
  font-size: 16px;
}

.prelabels-container {
  position: relative;
  user-select: none;
  padding: 10px;
}

.prelabels-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 12px;
}

.prelabel-card {
  background: white;
  border: 2px solid #e8e8e8;
  border-radius: 8px;
  padding: 8px;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
  width: 100%;
  min-height: 150px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.prelabel-card:hover,
.prelabel-card.high-confidence:hover,
.prelabel-card.medium-confidence:hover,
.prelabel-card.low-confidence:hover,
.prelabel-card.confirmed:hover,
.prelabel-card.skipped:hover,
.prelabel-card:not(.confirmed):not(.skipped):hover {
  border-color: #1890ff !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(24, 144, 255, 0.25);
}

.prelabel-card.selected {
  border: 3px dashed #1890ff !important;
  background: #e6f7ff !important;
  box-shadow: 0 0 12px rgba(24, 144, 255, 0.4);
  transform: scale(1.02);
}

.prelabel-card.confirmed {
  background: #f6ffed;
  /* 边框颜色保持置信度的颜色 */
}

.prelabel-card.skipped {
  border-color: #d9d9d9;
  background: #fafafa;
  opacity: 0.7;
}

.prelabel-card.skipped.selected {
  opacity: 1;
  background: #e6f7ff !important;
  border: 3px dashed #1890ff !important;
}

.prelabel-card.high-confidence {
  border-color: #52c41a !important;
}

.prelabel-card.medium-confidence {
  border-color: #faad14 !important;
}

.prelabel-card.low-confidence {
  border-color: #ff4d4f !important;
}

/* 待确认状态边框统一灰色，不受置信度影响 */
.prelabel-card:not(.confirmed):not(.skipped) {
  border-color: #d9d9d9 !important;
}

/* 选中状态下覆盖置信度颜色，统一为蓝色虚线边框和蓝色背景 */
.prelabel-card.selected.high-confidence,
.prelabel-card.selected.medium-confidence,
.prelabel-card.selected.low-confidence,
.prelabel-card.confirmed.selected,
.prelabel-card.confirmed.selected.high-confidence,
.prelabel-card.confirmed.selected.medium-confidence,
.prelabel-card.confirmed.selected.low-confidence {
  border: 3px dashed #1890ff !important;
  background: #e6f7ff !important;
}

.image-wrapper {
  width: 80px;
  height: 80px;
  margin: 0 auto 8px;
  aspect-ratio: 1;
  background: #f5f5f5;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
}

.image-wrapper img {
  width: 100%; 
  height: 100%; 

  object-fit: contain;
}

.info-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 12px;
  pointer-events: none;
}

.status-tag {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 500;
}

.status-tag.confirmed {
  background: #f6ffed;
  color: #52c41a;
}

.status-tag.skipped {
  background: #f5f5f5;
  color: #999;
}

.status-tag.pending {
  background: #e6f7ff;
  color: #1890ff;
}

.confidence-badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 500;
}

.confidence-badge.high {
  background: #f6ffed;
  color: #52c41a;
}

.confidence-badge.medium {
  background: #fffbe6;
  color: #faad14;
}

.confidence-badge.low {
  background: #fff1f0;
  color: #ff4d4f;
}

.predicted-char {
  text-align: center;
  font-size: 24px;
  font-weight: bold;
  color: #333;
  padding: 4px 0;
  pointer-events: none;
}

.predicted-char.group-red {
  color: #ff4d4f;
}

.predicted-char.group-yellow {
  color: #faad14;
}

.predicted-char.modified-char {
  font-weight: bold;
}

.selection-box {
  position: absolute;
  border: 2px solid #1890ff;
  background: rgba(24, 144, 255, 0.1);
  pointer-events: none;
  z-index: 100;
  box-sizing: border-box;
}

.selection-count {
  position: absolute;
  right: 4px;
  bottom: 4px;
  background: #1890ff;
  color: #fff;
  font-size: 12px;
  font-weight: bold;
  padding: 2px 6px;
  border-radius: 10px;
  line-height: 1.2;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 8px;
  min-width: 400px;
  max-width: 90vw;
  max-height: 90vh;
  overflow: auto;
}

.line-modal {
  max-width: 600px;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid #f0f0f0;
}

.modal-header h3 {
  margin: 0;
  font-size: 18px;
  color: #333;
}

.modal-body {
  padding: 20px;
}

.line-modal-body {
  padding: 20px;
}

.line-modal-hint {
  text-align: center;
  color: #999;
  font-size: 12px;
  margin-top: 12px;
}
</style>
