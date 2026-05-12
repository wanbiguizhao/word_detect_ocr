<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack" style="margin-right: 16px;">← 返回主页</a-button>
      <h1 class="title">多轮聚类标注</h1>
      <p class="desc">支持多次聚类，自动排除已标注字符</p>
    </div>

    <!-- 字符池统计 -->
    <div class="stats">
      <a-statistic title="总字符数" :value="stats.total" />
      <a-statistic title="已标注" :value="stats.labeled" />
      <a-statistic title="待标注" :value="stats.unlabeled" />
      <a-statistic title="聚类轮次" :value="rounds.length" />
    </div>

    <!-- 参数配置面板 -->
    <div class="params-panel">
      <a-collapse :activeKey="['params']" :bordered="false">
      <a-collapse-panel key="params" header="聚类参数配置">
        <a-form layout="inline">
          <a-form-item label="聚类方法">
            <a-select v-model:value="method" style="width: 120px;">
              <a-select-option value="hdbscan">HDBSCAN</a-select-option>
              <a-select-option value="kmeans">KMeans</a-select-option>
            </a-select>
          </a-form-item>
          
          <a-form-item v-if="method === 'hdbscan'" label="最小聚类大小">
            <input type="number" v-model.number="min_cluster_size" min="3" max="50" class="param-input" />
          </a-form-item>
          
          <a-form-item v-if="method === 'hdbscan'" label="核心点样本数">
            <input type="number" v-model.number="min_samples" min="1" max="20" class="param-input" />
          </a-form-item>
          
          <a-form-item v-if="method === 'hdbscan'" label="最大聚类大小">
            <input type="number" v-model.number="max_cluster_size" min="10" max="200" class="param-input" />
          </a-form-item>
          
          <a-form-item v-if="method === 'kmeans'" label="目标聚类数">
            <input type="number" v-model.number="n_clusters" min="2" max="100" class="param-input" />
          </a-form-item>
        </a-form>
        <a-button @click="showParams" style="margin-top: 12px;">显示当前参数</a-button>
      </a-collapse-panel>
    </a-collapse>
    </div>

    <!-- 操作区域 -->
    <div class="toolbar">
      <div class="new-round-form">
        <a-input-number v-model:value="newRoundClusters" :min="1" :max="1000" style="width: 120px;" />
        <a-input v-model:value="newRoundDesc" placeholder="轮次描述" style="width: 200px;" />
        <a-button type="primary" @click="startNewRound" :loading="isStarting">
          启动新轮聚类
        </a-button>
      </div>
      <a-button @click="refreshData">刷新数据</a-button>
    </div>

    <!-- 轮次选择 -->
    <div class="round-tabs">
      <a-radio-group v-model:value="activeRound" @change="handleRoundChange">
        <a-radio-button v-for="round in rounds" :key="round.round" :value="round.round">
          第{{ round.round }}轮
        </a-radio-button>
      </a-radio-group>
    </div>

    <!-- 当前轮次信息 -->
    <div v-if="currentRound" class="round-info-card">
      <div class="round-info-header">
        <span class="round-title">第 {{ currentRound.round }} 轮聚类</span>
        <span class="round-date">{{ formatDate(currentRound.date) }}</span>
      </div>
      <div class="round-info-body">
        <span>描述: {{ currentRound.description }}</span>
        <span>聚类数: {{ currentRound.n_clusters }}</span>
        <span>字符数: {{ currentRound.total_chars }}</span>
      </div>
    </div>

    <!-- 聚类列表 -->
    <a-table
      v-if="clusters.length > 0"
      :data-source="clusters"
      row-key="cluster_id"
      bordered
      :pagination="{ pageSize: 20 }"
    >
      <a-table-column title="聚类ID" data-index="cluster_id" width="100" />
      <a-table-column title="状态" width="100">
        <template #default="{ record }">
          <a-tag :color="getStatusColor(record.status)">
            {{ getStatusText(record.status) }}
          </a-tag>
        </template>
      </a-table-column>
      <a-table-column title="汉字" width="200">
        <template #default="{ record }">
          <template v-if="record.char_counts && Object.keys(record.char_counts).length > 0">
            <span 
              v-for="(count, char) in record.char_counts" 
              :key="char" 
              class="char-tag"
            >
              {{ char }}({{ count }})
            </span>
          </template>
          <template v-else>
            <span class="chars-display">{{ record.char || '?' }}</span>
          </template>
        </template>
      </a-table-column>
      <a-table-column title="字符数量" width="120">
        <template #default="{ record }">
          {{ record.labeled_count || 0 }}/{{ record.char_count }}
        </template>
      </a-table-column>
      <a-table-column title="操作" width="150">
        <template #default="{ record }">
          <a-button 
            :type="activeClusterId === record.cluster_id ? 'default' : 'primary'" 
            size="small" 
            @click="goClusterLabel(record.cluster_id)"
            :style="activeClusterId === record.cluster_id ? 'background: #faad14; border-color: #faad14; color: white;' : ''"
          >
            {{ activeClusterId === record.cluster_id ? '标注中' : '标注' }}
          </a-button>
          <a-button size="small" @click="skipCluster(record.cluster_id)" danger style="margin-left: 8px;">
            跳过
          </a-button>
        </template>
      </a-table-column>
    </a-table>

    <div v-else class="empty-state">
      <div class="empty-icon">📊</div>
      <p>暂无聚类数据</p>
      <p>点击上方按钮启动聚类</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()

const API_BASE = '/api/mc'

const stats = ref({
  total: 0,
  labeled: 0,
  unlabeled: 0
})

const rounds = ref([])
const activeRound = ref(null)
const clusters = ref([])
const currentRound = ref(null)

const newRoundClusters = ref(100)
const newRoundDesc = ref('')
const isStarting = ref(false)
const activeClusterId = ref(null)

// 聚类参数配置 - 使用单独的 ref
const method = ref('hdbscan')
const min_cluster_size = ref(5)
const min_samples = ref(2)
const max_cluster_size = ref(100)
const n_clusters = ref(20)

const formatDate = (dateStr) => {
  const date = new Date(dateStr)
  return date.toLocaleString('zh-CN')
}

const getStatusColor = (status) => {
  switch (status) {
    case 'labeled': return 'green'
    case 'skipped': return 'red'
    default: return 'default'
  }
}

const getStatusText = (status) => {
  switch (status) {
    case 'labeled': return '已标注'
    case 'skipped': return '已跳过'
    default: return '未标注'
  }
}

const getCharImage = (charId) => {
  return `/api/image/pdf_chars/${charId}`
}

const fetchStats = async () => {
  try {
    const res = await axios.get(`${API_BASE}/char-pool`)
    if (res.data.code === 0) {
      stats.value = res.data.data
    }
  } catch (e) {
    console.error('获取统计失败:', e)
  }
}

const fetchRounds = async () => {
  try {
    const res = await axios.get(`${API_BASE}/rounds`)
    if (res.data.code === 0) {
      rounds.value = res.data.data.rounds || []
      if (rounds.value.length > 0 && !activeRound.value) {
        activeRound.value = rounds.value[rounds.value.length - 1].round
      }
    }
  } catch (e) {
    console.error('获取轮次失败:', e)
  }
}

const fetchClusters = async (roundNum) => {
  if (!roundNum) return
  
  try {
    const res = await axios.get(`${API_BASE}/rounds/${roundNum}/clusters`)
    if (res.data.code === 0) {
      clusters.value = res.data.clusters || []
    }
  } catch (e) {
    console.error('获取聚类失败:', e)
  }
}

const fetchRoundDetail = async (roundNum) => {
  if (!roundNum) return
  
  try {
    const res = await axios.get(`${API_BASE}/rounds/${roundNum}`)
    if (res.data.code === 0) {
      currentRound.value = res.data
    }
  } catch (e) {
    console.error('获取轮次详情失败:', e)
  }
}

const showParams = () => {
  console.log('[DEBUG] method:', method.value)
  console.log('[DEBUG] min_cluster_size:', min_cluster_size.value)
  console.log('[DEBUG] min_samples:', min_samples.value)
  console.log('[DEBUG] max_cluster_size:', max_cluster_size.value)
  console.log('[DEBUG] n_clusters:', n_clusters.value)
  
  alert(`当前参数:\n\nmethod: ${method.value}\nmin_cluster_size: ${min_cluster_size.value}\nmin_samples: ${min_samples.value}\nmax_cluster_size: ${max_cluster_size.value}\nn_clusters: ${n_clusters.value}`)
}

const startNewRound = async () => {
  if (isStarting.value) return
  
  isStarting.value = true
  try {
    console.log('[Frontend] 准备发送的参数:')
    console.log('  method:', method.value)
    console.log('  max_cluster_size:', max_cluster_size.value)
    
    const params = {
      description: newRoundDesc.value || `第${rounds.value.length + 1}轮聚类`,
      method: method.value
    }
    
    if (method.value === 'hdbscan') {
      params.min_cluster_size = min_cluster_size.value
      params.min_samples = min_samples.value
      params.max_cluster_size = max_cluster_size.value
    } else if (method.value === 'kmeans') {
      params.n_clusters = n_clusters.value
    }
    
    console.log('[Frontend] 将要发送的参数:', params)
    
    const res = await axios.post(`${API_BASE}/rounds`, params)
    if (res.data.code === 0) {
      alert(`第${res.data.round}轮聚类已启动！`)
      newRoundClusters.value = 100
      newRoundDesc.value = ''
      await refreshData()
    } else {
      alert(res.data.msg || '启动失败')
    }
  } catch (e) {
    console.error('启动聚类失败:', e)
    alert('启动聚类失败')
  } finally {
    isStarting.value = false
  }
}

const handleRoundChange = (e) => {
  const roundNum = e.target.value
  activeRound.value = roundNum
  fetchClusters(roundNum)
  fetchRoundDetail(roundNum)
}

const goClusterLabel = (clusterId) => {
  activeClusterId.value = clusterId
  const url = `${window.location.origin}/mc-label/${activeRound.value}/${clusterId}`
  window.open(url, '_blank')
}

const skipCluster = async (clusterId) => {
  try {
    await axios.post(`${API_BASE}/rounds/${activeRound.value}/clusters/${clusterId}/skip`)
    await fetchClusters(activeRound.value)
  } catch (e) {
    console.error('跳过聚类失败:', e)
  }
}

const goBack = () => {
  router.push('/')
}

const refreshData = async () => {
  await fetchStats()
  await fetchRounds()
  if (activeRound.value) {
    await fetchClusters(activeRound.value)
    await fetchRoundDetail(activeRound.value)
  }
}

onMounted(() => {
  refreshData()
})
</script>

<style scoped>
.page-container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
}

.title {
  font-size: 24px;
  margin: 0;
}

.desc {
  color: #666;
  margin: 0 0 0 auto;
}

.stats {
  display: flex;
  gap: 40px;
  margin-bottom: 20px;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
}

.params-panel {
  margin-bottom: 20px;
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  overflow: hidden;
}

.params-panel :deep(.ant-collapse) {
  background: transparent;
  border: none;
}

.params-panel :deep(.ant-collapse-item) {
  border-bottom: 1px solid #e8e8e8;
}

.params-panel :deep(.ant-collapse-item:last-child) {
  border-bottom: none;
}

.params-panel :deep(.ant-collapse-header) {
  padding: 12px 16px;
  font-weight: bold;
  background: #fafafa;
}

.params-panel :deep(.ant-collapse-content) {
  padding: 16px;
}

.param-input {
  width: 100px;
  padding: 4px 11px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  font-size: 14px;
}

.param-input:focus {
  outline: none;
  border-color: #1890ff;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.new-round-form {
  display: flex;
  gap: 12px;
  align-items: center;
}

.round-tabs {
  margin-bottom: 20px;
}

.round-info-card {
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 20px;
}

.round-info-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.round-title {
  font-weight: bold;
}

.round-date {
  color: #999;
  font-size: 14px;
}

.round-info-body {
  display: flex;
  gap: 24px;
  color: #666;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  background: #f8f9fa;
  border-radius: 12px;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.chars-display {
  font-size: 16px;
  font-weight: bold;
}

.char-tag {
  display: inline-block;
  background: #f0f5ff;
  color: #1890ff;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 14px;
  margin-right: 4px;
  margin-bottom: 4px;
}
</style>