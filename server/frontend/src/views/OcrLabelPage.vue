<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack" style="margin-right: 16px;">← 返回主页</a-button>
      <h1 class="title">汉字OCR标注</h1>
      <p class="desc">聚类标注 | 数据来源：clusters/hog_clusters.json</p>
    </div>

    <div class="toolbar">
      <div class="sort-controls">
        <span>排序方式：</span>
        <a-radio-group v-model:value="sortBy" @change="handleSort">
          <a-radio value="count">图片数量</a-radio>
          <a-radio value="labeled">已标记数量</a-radio>
          <a-radio value="unlabeled">未标记数量</a-radio>
          <a-radio value="chaos">混乱度</a-radio>
          <a-radio value="confidence">置信度</a-radio>
        </a-radio-group>
      </div>
      <div class="filter-controls">
        <span>过滤：</span>
        <a-select v-model:value="filterStatus" style="width: 120px;">
          <a-select-option value="all">全部</a-select-option>
          <a-select-option value="unlabeled">未标注</a-select-option>
          <a-select-option value="labeled">已标注</a-select-option>
          <a-select-option value="skipped">暂不标记</a-select-option>
        </a-select>
      </div>
      <a-button @click="loadData" :loading="loading">刷新数据</a-button>
    </div>

    <a-table
      :data-source="displayClusters"
      row-key="clusterId"
      bordered
      :pagination="{ pageSize: 20 }"
      :row-class-name="(record) => record.clusterId === highlightedClusterId ? 'highlighted-row' : ''"
    >
      <a-table-column title="聚类ID" data-index="clusterId" width="100" />
      <a-table-column title="别名" width="150">
        <template #default="{ record }">
          <a-input
            v-model:value="record.alias"
            placeholder="设置别名"
            @blur="saveAlias(record)"
          />
        </template>
      </a-table-column>
      <a-table-column title="状态" width="100">
        <template #default="{ record }">
          <a-tag :color="getStatusColor(record.status)">
            {{ getStatusText(record.status) }}
          </a-tag>
        </template>
      </a-table-column>
      <a-table-column title="汉字(数量)" width="150">
        <template #default="{ record }">
          <span class="chars-display">{{ record.charsDisplay || '?' }}</span>
        </template>
      </a-table-column>
      <a-table-column title="混乱度" width="100">
        <template #default="{ record }">
          <a-progress
            :percent="record.chaos * 100"
            :format="percent => `${(percent).toFixed(1)}%`"
            :status="record.chaos > 0.5 ? 'exception' : record.chaos > 0.2 ? 'normal' : 'success'"
            size="small"
          />
        </template>
      </a-table-column>
      <a-table-column title="置信度" width="100">
        <template #default="{ record }">
          <a-tag :color="record.confidence ? (record.confidence >= 0.8 ? 'green' : record.confidence >= 0.5 ? 'orange' : 'red') : 'default'">
            {{ record.confidence ? `${(record.confidence * 100).toFixed(0)}%` : '-' }}
          </a-tag>
        </template>
      </a-table-column>
      <a-table-column title="图片数量" data-index="totalCount" width="100" />
      <a-table-column title="已标记" data-index="labeledCount" width="100">
        <template #default="{ record }">
          <a-tag color="green">{{ record.labeledCount }}</a-tag>
        </template>
      </a-table-column>
      <a-table-column title="未标记" data-index="unlabeledCount" width="100">
        <template #default="{ record }">
          <a-tag color="orange">{{ record.unlabeledCount }}</a-tag>
        </template>
      </a-table-column>
      <a-table-column title="操作" width="120">
        <template #default="{ record }">
          <a-button type="primary" size="small" @click="goClusterLabel(record.clusterId)">
            标注
          </a-button>
        </template>
      </a-table-column>
    </a-table>

    <div class="stats">
      <a-statistic title="总聚类数" :value="clusters.length" />
      <a-statistic title="已标记聚类" :value="labeledClusters" />
      <a-statistic title="未标记聚类" :value="unlabeledClusters" />
      <a-statistic title="总汉字数" :value="totalChars" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()

const loading = ref(false)
const sortBy = ref('chaos')
const filterStatus = ref('all')
const clusters = ref([])
const labels = ref({})
const highlightedClusterId = ref(null)

const displayClusters = computed(() => {
  let filtered = [...clusters.value]
  
  if (filterStatus.value !== 'all') {
    filtered = filtered.filter(c => c.status === filterStatus.value)
  }
  
  return filtered.sort((a, b) => {
    switch (sortBy.value) {
      case 'count':
        return b.totalCount - a.totalCount
      case 'labeled':
        return b.labeledCount - a.labeledCount
      case 'unlabeled':
        return b.unlabeledCount - a.unlabeledCount
      case 'chaos':
        return b.chaos - a.chaos
      case 'confidence':
        const confA = a.confidence || 0
        const confB = b.confidence || 0
        return confB - confA
      default:
        return 0
    }
  })
})

const labeledClusters = computed(() => clusters.value.filter(c => c.labeledCount > 0).length)
const unlabeledClusters = computed(() => clusters.value.filter(c => c.labeledCount === 0).length)
const totalChars = computed(() => clusters.value.reduce((sum, c) => sum + c.totalCount, 0))

const calculateChaos = (chars) => {
  if (!chars || chars.length === 0) return 0

  const charCount = {}
  chars.forEach(char => {
    const suggested = char.lineage?.suggested_char || char.lineage?.char || 'unknown'
    charCount[suggested] = (charCount[suggested] || 0) + 1
  })

  const total = chars.length
  let entropy = 0

  Object.values(charCount).forEach(count => {
    const p = count / total
    if (p > 0) {
      entropy -= p * Math.log2(p)
    }
  })

  const maxEntropy = Math.log2(Object.keys(charCount).length || 1)
  return maxEntropy > 0 ? entropy / maxEntropy : 0
}

const loadData = async () => {
  loading.value = true
  try {
    const [clustersRes, labelsRes] = await Promise.all([
      axios.get('/api/clusters'),
      axios.get('/api/cluster-labels')
    ])

    const clustersData = clustersRes.data.clusters || {}
    const labelsData = labelsRes.data.data || {}

    const clustersList = Object.entries(clustersData).map(([clusterId, chars]) => {
      const charList = Array.isArray(chars) ? chars : []
      const clusterLabels = labelsData[clusterId] || {}
      const charLabels = clusterLabels.char_labels || {}
      const charsMap = clusterLabels.chars || {}

      let labeledCount = 0
      let unlabeledCount = 0

      charList.forEach((char, idx) => {
        const labelKey = String(idx)
        if (charLabels[labelKey]?.char) {
          labeledCount++
        } else {
          unlabeledCount++
        }
      })

      const charsDisplay = Object.entries(charsMap).length > 0
        ? Object.entries(charsMap)
            .sort((a, b) => b[1] - a[1])
            .map(([char, count]) => `${char}(${count})`)
            .join(', ')
        : (clusterLabels.char ? `${clusterLabels.char}(${labeledCount})` : '')

      const status = clusterLabels.status || (labeledCount > 0 ? 'labeled' : 'unlabeled')

      return {
        clusterId: parseInt(clusterId),
        chars: charList,
        totalCount: charList.length,
        labeledCount,
        unlabeledCount,
        chaos: calculateChaos(charList),
        confidence: clusterLabels.confidence || null,
        charsDisplay,
        alias: clusterLabels.alias || '',
        status
      }
    })

    clusters.value = clustersList
    labels.value = labelsData
  } catch (err) {
    console.error('加载数据失败:', err)
  } finally {
    loading.value = false
  }
}

const getStatusColor = (status) => {
  switch (status) {
    case 'labeled':
      return 'green'
    case 'skipped':
      return 'orange'
    default:
      return 'default'
  }
}

const getStatusText = (status) => {
  switch (status) {
    case 'labeled':
      return '已标注'
    case 'skipped':
      return '暂不标记'
    default:
      return '未标注'
  }
}

const handleSort = () => {
}

const saveAlias = async (record) => {
  try {
    await axios.post('/api/cluster-labels/save', {
      clusterId: record.clusterId,
      alias: record.alias
    })
  } catch (err) {
    console.error('保存别名失败:', err)
  }
}

const goClusterLabel = (clusterId) => {
  highlightedClusterId.value = parseInt(clusterId)
  const dataset = localStorage.getItem('selectedDataset') || 'pdf01'
  window.open(`/ocr-label/${clusterId}?dataset=${encodeURIComponent(dataset)}`, '_blank')
}

const goBack = () => {
  router.push('/')
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.page-container { padding: 30px; max-width: 1400px; margin: 0 auto; }
.header { margin-bottom: 24px; display: flex; align-items: center; }
.title { margin: 0 16px 0 0; }
.desc { color: #666; margin: 0; }
.toolbar { display: flex; justify-content: space-between; margin-bottom: 16px; }
.sort-controls { display: flex; align-items: center; gap: 12px; }
.suggested-char { font-size: 18px; font-weight: bold; }
.chars-display { font-size: 16px; font-weight: bold; color: #1890ff; }
.stats {
  display: flex;
  justify-content: space-around;
  margin-top: 32px;
  padding: 24px;
  background: #f5f5f5;
  border-radius: 8px;
}
.highlighted-row {
  background: #e6f7ff !important;
  transition: background 0.3s ease;
}
:deep(.highlighted-row) {
  background: #e6f7ff !important;
}
:deep(.highlighted-row td) {
  background: #e6f7ff !important;
}
</style>