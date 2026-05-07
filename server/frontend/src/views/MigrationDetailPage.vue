<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack" style="margin-right: 16px;">← 返回列表</a-button>
      <h1 class="title">标注详情 - '{{ selectedChar }}'</h1>
      <p class="desc">匹配结果共 {{ matches.length }} 张图片</p>
    </div>

    <div class="stats-row">
      <a-statistic title="高置信度匹配" :value="highConfidenceCount" />
      <a-statistic title="待确认匹配" :value="pendingCount" />
      <a-statistic title="已选中" :value="selectedImages.length" />
      <a-statistic title="已标注" :value="labeledCount" />
    </div>

    <div class="toolbar">
      <div class="left-actions">
        <a-button @click="selectAll">全选</a-button>
        <a-button @click="selectHighConfidence">全选高置信度</a-button>
        <a-button @click="clearSelection">清除选择</a-button>
        <a-button @click="toggleSelectMode">{{ selectMode ? '退出选择模式' : '进入选择模式' }}</a-button>
      </div>
      <div class="right-actions">
        <a-button
          type="primary"
          @click="batchLabel"
          :disabled="selectedImages.length === 0"
          :loading="loading"
        >
          批量标注选中 ({{ selectedImages.length }})
        </a-button>
      </div>
    </div>

    <div v-if="matches.length > 0" class="matches-grid" @contextmenu.prevent>
      <div
        v-for="(match, index) in sortedMatches"
        :key="match.char_id"
        class="match-card"
        :class="{
          selected: selectedImages.includes(match.char_id),
          labeled: match.status === 'labeled',
          'high-confidence': match.similarity >= 0.8,
          'low-confidence': match.similarity < 0.5
        }"
        @click="handleCardClick(match)"
        @contextmenu.prevent="showContextMenu($event, match, index)"
      >
        <div class="image-wrapper">
          <img :src="getImageUrl(match.image_filename || match.image_path)" :alt="match.char_id" />
          <div v-if="match.status === 'labeled'" class="labeled-overlay">
            <span>已标注</span>
          </div>
        </div>
        <div class="match-info">
          <div class="similarity-badge" :class="getSimilarityClass(match.similarity)">
            {{ (match.similarity * 100).toFixed(0) }}%
          </div>
          <span class="cluster-info">聚类{{ match.cluster_id }}</span>
        </div>
        <!-- 单个标注输入框 -->
        <div class="label-input-area">
          <a-input
            :value="match.status === 'labeled' ? match.label : (pendingLabels[match.char_id] || '')"
            :disabled="match.status === 'labeled'"
            placeholder="标注"
            style="width: 60px;"
            @input="updatePendingLabel(match.char_id, $event)"
            @blur="saveSingleLabel(match)"
            @click.stop
          />
        </div>
        <div v-if="selectMode" class="checkbox" :class="{ checked: selectedImages.includes(match.char_id) }">
          <span v-if="selectedImages.includes(match.char_id)">✓</span>
        </div>
      </div>
    </div>

    <div v-else class="empty-state">
      <div class="empty-icon">📭</div>
      <p>暂无匹配结果</p>
      <p class="hint">该汉字在目标数据集中没有找到匹配图片</p>
    </div>

    <!-- 右键菜单 -->
    <a-modal
      v-model:open="showContextMenuModal"
      title="操作菜单"
      :footer="null"
      width="300px"
    >
      <div class="context-menu">
        <a-list :data-source="contextMenuItems">
          <template #renderItem="{ item }">
            <a-list-item @click="handleContextMenuClick(item.key)">
              <component :is="item.icon" style="margin-right: 8px;" />
              <span>{{ item.label }}</span>
            </a-list-item>
          </template>
        </a-list>
      </div>
    </a-modal>

    <!-- 单个标注弹窗 -->
    <a-modal
      v-model:open="showSingleLabelModal"
      title="标注单个字符"
      :footer="null"
      width="400px"
    >
      <div v-if="currentMatch" class="single-label-form">
        <div class="preview-section">
          <img :src="getImageUrl(currentMatch.image_filename || currentMatch.image_path)" :alt="currentMatch.char_id" />
        </div>
        <a-form :model="labelForm" layout="vertical">
          <a-form-item label="标注汉字">
            <a-input v-model="labelForm.char" :value="selectedChar" />
          </a-form-item>
          <a-form-item>
            <a-button type="primary" @click="submitSingleLabel">确认标注</a-button>
            <a-button @click="showSingleLabelModal = false" style="margin-left: 8px;">取消</a-button>
          </a-form-item>
        </a-form>
      </div>
    </a-modal>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { EditOutlined, EyeOutlined, DeleteOutlined, CheckOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'
import axios from 'axios'

const route = useRoute()
const router = useRouter()

const selectedChar = ref('')
const matches = ref([])
const selectedImages = ref([])
const selectMode = ref(true)
const loading = ref(false)
const pendingLabels = ref({})

const showContextMenuModal = ref(false)
const showSingleLabelModal = ref(false)
const currentMatch = ref(null)
const contextMenuIndex = ref(null)

const labelForm = ref({
  char: ''
})

const contextMenuItems = [
  { key: 'label', label: '标注此字符', icon: EditOutlined },
  { key: 'view', label: '查看详情', icon: EyeOutlined },
  { key: 'remove', label: '从列表移除', icon: DeleteOutlined }
]

const highConfidenceCount = computed(() => matches.value.filter(m => m.similarity >= 0.8).length)
const pendingCount = computed(() => matches.value.filter(m => m.similarity < 0.5).length)
const labeledCount = computed(() => matches.value.filter(m => m.status === 'labeled').length)

const sortedMatches = computed(() => {
  return [...matches.value].sort((a, b) => {
    if (a.status === 'labeled' && b.status !== 'labeled') return 1
    if (a.status !== 'labeled' && b.status === 'labeled') return -1
    return b.similarity - a.similarity
  })
})

const getImageUrl = (imagePath) => {
  if (!imagePath) return ''
  const filename = imagePath.split('/').pop().split('\\').pop()
  return `/api/char-images/${filename}`
}

const getSimilarityClass = (similarity) => {
  if (similarity >= 0.8) return 'high'
  if (similarity >= 0.5) return 'medium'
  return 'low'
}

const goBack = () => {
  router.push('/migration-list')
}

const loadMatches = async () => {
  if (!selectedChar.value) return
  
  loading.value = true
  try {
    const res = await axios.get(`/api/migration/image-matches/${selectedChar.value}`)
    if (res.data.code === 0) {
      matches.value = res.data.matches || []
    }
  } catch (err) {
    console.error('加载匹配结果失败:', err)
  } finally {
    loading.value = false
  }
}

const handleCardClick = (match) => {
  if (match.status === 'labeled') return
  
  if (selectMode.value) {
    toggleSelect(match.char_id)
  }
}

const toggleSelect = (charId) => {
  const idx = selectedImages.value.indexOf(charId)
  if (idx === -1) {
    selectedImages.value.push(charId)
  } else {
    selectedImages.value.splice(idx, 1)
  }
}

const selectAll = () => {
  const unlabeled = matches.value.filter(m => m.status !== 'labeled')
  selectedImages.value = unlabeled.map(m => m.char_id)
}

const selectHighConfidence = () => {
  const highConfidence = matches.value.filter(m => m.status !== 'labeled' && m.similarity >= 0.8)
  selectedImages.value = highConfidence.map(m => m.char_id)
}

const clearSelection = () => {
  selectedImages.value = []
}

const toggleSelectMode = () => {
  selectMode.value = !selectMode.value
  if (!selectMode.value) {
    clearSelection()
  }
}

const showContextMenu = (event, match, index) => {
  currentMatch.value = match
  contextMenuIndex.value = index
  showContextMenuModal.value = true
}

const handleContextMenuClick = (key) => {
  showContextMenuModal.value = false
  
  if (!currentMatch.value) return
  
  switch (key) {
    case 'label':
      labelForm.value.char = selectedChar.value
      showSingleLabelModal.value = true
      break
    case 'view':
      viewDetail(currentMatch.value)
      break
    case 'remove':
      removeFromList(currentMatch.value)
      break
  }
}

const viewDetail = (match) => {
  console.log('查看详情:', match)
}

const removeFromList = (match) => {
  matches.value = matches.value.filter(m => m.char_id !== match.char_id)
}

const submitSingleLabel = async () => {
  if (!labelForm.value.char || !currentMatch.value) return
  
  loading.value = true
  try {
    const res = await axios.post('/api/migration/batch-label', {
      char: labelForm.value.char,
      charIds: [currentMatch.value.char_id]
    })
    
    if (res.data.code === 0) {
      message.success('标注成功')
      showSingleLabelModal.value = false
      currentMatch.value = null
      loadMatches()
    } else {
      message.error(res.data.msg || '标注失败')
    }
  } catch (err) {
    console.error('标注失败:', err)
    message.error('标注失败')
  } finally {
    loading.value = false
  }
}

const updatePendingLabel = (charId, event) => {
  const value = event.target.value
  if (value) {
    pendingLabels.value[charId] = value
  } else {
    delete pendingLabels.value[charId]
  }
}

const saveSingleLabel = async (match) => {
  const char = pendingLabels.value[match.char_id]
  if (!char) return
  
  loading.value = true
  try {
    const res = await axios.post('/api/migration/batch-label', {
      char: char,
      charIds: [match.char_id]
    })
    
    if (res.data.code === 0) {
      message.success('标注成功')
      delete pendingLabels.value[match.char_id]
      loadMatches()
    } else {
      message.error(res.data.msg || '标注失败')
    }
  } catch (err) {
    console.error('标注失败:', err)
    message.error('标注失败')
  } finally {
    loading.value = false
  }
}

const batchLabel = async () => {
  if (selectedImages.value.length === 0) return
  
  loading.value = true
  try {
    const res = await axios.post('/api/migration/batch-label', {
      char: selectedChar.value,
      charIds: selectedImages.value
    })
    
    if (res.data.code === 0) {
      message.success(`成功标注 ${selectedImages.value.length} 张图片`)
      selectedImages.value = []
      loadMatches()
    } else {
      message.error(res.data.msg || '批量标注失败')
    }
  } catch (err) {
    console.error('批量标注失败:', err)
    message.error('批量标注失败')
  } finally {
    loading.value = false
  }
}

watch(() => route.params.char, (newChar) => {
  if (newChar) {
    selectedChar.value = newChar
    selectedImages.value = []
    loadMatches()
  }
})

onMounted(() => {
  selectedChar.value = route.params.char || ''
  loadMatches()
})
</script>

<style scoped>
.page-container {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}

.header {
  display: flex;
  align-items: center;
  margin-bottom: 24px;
}

.title {
  margin: 0 16px 0 0;
  font-size: 24px;
}

.desc {
  margin: 0;
  color: #666;
}

.stats-row {
  display: flex;
  gap: 24px;
  margin-bottom: 16px;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding: 16px;
  background: #f5f5f5;
  border-radius: 8px;
}

.left-actions {
  display: flex;
  gap: 8px;
}

.right-actions {
  display: flex;
  gap: 8px;
}

.matches-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 16px;
}

.match-card {
  position: relative;
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  cursor: pointer;
  transition: all 0.2s ease;
}

.match-card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
}

.match-card.selected {
  box-shadow: 0 0 0 3px #1890ff;
}

.match-card.labeled {
  border: 2px solid #52c41a;
  background: #f6ffed;
}

.match-card.labeled .image-wrapper {
  background: #e6f7ff;
}

.match-card.high-confidence {
  border: 2px solid #52c41a;
}

.match-card.low-confidence {
  border: 2px solid #ff4d4f;
}

.image-wrapper {
  position: relative;
  width: 100%;
  padding-top: 100%;
  background: #f5f5f5;
}

.image-wrapper img {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.labeled-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(82, 196, 26, 0.8);
  color: #fff;
  font-weight: bold;
}

.match-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
}

.similarity-badge {
  font-size: 12px;
  font-weight: bold;
  padding: 2px 8px;
  border-radius: 4px;
}

.similarity-badge.high {
  background: #f6ffed;
  color: #52c41a;
}

.similarity-badge.medium {
  background: #fff7e6;
  color: #fa8c16;
}

.similarity-badge.low {
  background: #fff1f0;
  color: #ff4d4f;
}

.cluster-info {
  font-size: 12px;
  color: #999;
}

.label-input-area {
  padding: 0 8px 8px;
}

.label-input-area :deep(.ant-input) {
  text-align: center;
  font-size: 16px;
  font-weight: bold;
}

.checkbox {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 24px;
  height: 24px;
  border: 2px solid #d9d9d9;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.9);
  font-size: 14px;
  color: #1890ff;
}

.checkbox.checked {
  background: #1890ff;
  border-color: #1890ff;
  color: #fff;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 0;
  color: #999;
}

.empty-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.hint {
  font-size: 14px;
  color: #bbb;
}

.context-menu {
  padding: 8px 0;
}

.single-label-form {
  padding: 16px;
}

.preview-section {
  text-align: center;
  margin-bottom: 16px;
}

.preview-section img {
  max-width: 200px;
  max-height: 200px;
  border-radius: 8px;
}
</style>