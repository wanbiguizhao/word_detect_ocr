<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack" style="margin-right: 16px;">← 返回主页</a-button>
      <h1 class="title">字符切割标注</h1>
      <p class="desc">主动学习样本 | 数据来源：top_annotate_samples.json</p>
    </div>

    <!-- 统计面板 -->
    <div class="stats-panel">
      <div class="stat-card">
        <a-statistic title="总样本数" :value="stats.total_samples" />
      </div>
      <div class="stat-card">
        <a-statistic title="已标注" :value="stats.annotated_count" />
      </div>
      <div class="stat-card">
        <a-statistic title="暂不标注" :value="stats.postponed_count" />
      </div>
      <div class="stat-card">
        <a-statistic title="未标注" :value="stats.unannotated_count" />
      </div>
      <div class="stat-card progress-card">
        <a-progress :percent="stats.progress" :show-info="true" />
      </div>
    </div>

    <!-- 快捷操作 -->
    <div class="quick-actions">
      <a-button type="primary" size="large" @click="goNextLabel" :disabled="!nextSample">
        ⚡ 快速标注下一个
      </a-button>
      <a-button type="default" size="large" @click="loadPriorityQueue">
        🔄 刷新优先级队列
      </a-button>
    </div>

    <!-- 标签切换 -->
    <a-tabs v-model:activeKey="activeTab" @change="handleTabChange">
      <a-tab-pane key="list" tab="样本列表">
        <div style="margin-bottom: 16px;">
          <a-select
            placeholder="筛选标注状态"
            style="width: 180px"
            @change="handleSelectChange"
          >
            <a-select-option value="all">全部样本</a-select-option>
            <a-select-option value="pending">暂不标注</a-select-option>
            <a-select-option value="true">已标注</a-select-option>
            <a-select-option value="false">未标注</a-select-option>
          </a-select>
        </div>

        <a-table
          :data-source="paginatedList"
          row-key="id"
          bordered
          :pagination="pagination"
          @change="handlePageChange"
        >
          <a-table-column title="文件名称" data-index="image_name" width="300" />
          <a-table-column title="主动学习分数" width="150">
            <template #default="{ record }">
              {{ record.score.toFixed(4) }}
            </template>
          </a-table-column>
          <a-table-column title="标注状态" width="150">
            <template #default="{ record }">
              <a-tag v-if="record.is_postponed" color="purple">
                暂不标注
              </a-tag>
              <a-tag v-else-if="record.is_annotated" color="green">
                已标注
              </a-tag>
              <a-tag v-else color="orange">
                未标注
              </a-tag>
            </template>
          </a-table-column>
          <a-table-column title="上一次更新时间" width="200">
            <template #default="{ record }">
              {{ record.updated_at ? formatTime(record.updated_at) : '未更新' }}
            </template>
          </a-table-column>
          <a-table-column title="操作" width="160">
            <template #default="{ record }">
              <a-button type="primary" @click="goLabel(record.id)">
                进入标注
              </a-button>
            </template>
          </a-table-column>
        </a-table>
      </a-tab-pane>

      <a-tab-pane key="priority" tab="优先级队列">
        <a-table
          :data-source="priorityQueue"
          row-key="image_id"
          bordered
          :pagination="{ pageSize: 10 }"
        >
          <a-table-column title="优先级" width="80">
            <template #default="{ record, index }">
              <a-tag color="red" v-if="index === 0">🔥 TOP1</a-tag>
              <a-tag color="orange" v-else-if="index === 1">🔥 TOP2</a-tag>
              <a-tag color="gold" v-else-if="index === 2">🔥 TOP3</a-tag>
              <span v-else>{{ index + 1 }}</span>
            </template>
          </a-table-column>
          <a-table-column title="文件名称" data-index="image_name" width="300" />
          <a-table-column title="优先级分数" width="150">
            <template #default="{ record }">
              <a-tag :color="getPriorityColor(record.priority_score)">
                {{ record.priority_score.toFixed(4) }}
              </a-tag>
            </template>
          </a-table-column>
          <a-table-column title="主动学习分数" width="150">
            <template #default="{ record }">
              {{ record.al_score.toFixed(4) }}
            </template>
          </a-table-column>
          <a-table-column title="操作" width="160">
            <template #default="{ record }">
              <a-button type="primary" @click="goLabel(record.image_id)">
                标注
              </a-button>
            </template>
          </a-table-column>
        </a-table>
      </a-tab-pane>
    </a-tabs>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import { Statistic as aStatistic, Progress as aProgress, Tabs as aTabs, TabPane as aTabPane } from 'ant-design-vue'

const router = useRouter()
const allData = ref([])
const filterAnnotated = ref("all")
const activeTab = ref("list")
const priorityQueue = ref([])
const nextSample = ref(null)
const stats = ref({
  total_samples: 0,
  annotated_count: 0,
  postponed_count: 0,
  unannotated_count: 0,
  progress: 0
})

const pagination = ref({
  pageSize: 10,
  current: 1,
  showSizeChanger: true,
  pageSizeOptions: [10, 20, 50, 100],
  total: 0
})

let eventSource = null

const paginatedList = computed(() => {
  const start = (pagination.value.current - 1) * pagination.value.pageSize
  const end = start + pagination.value.pageSize
  return allData.value.slice(start, end)
})

const handleSelectChange = (selectedValue) => {
  filterAnnotated.value = selectedValue
  pagination.value.current = 1
  loadList()
}

const handlePageChange = (page) => {
  pagination.value.current = page.current
  pagination.value.pageSize = page.pageSize
}

const handleTabChange = (key) => {
  if (key === 'priority') {
    loadPriorityQueue()
  }
}

const handleEvent = (event) => {
  try {
    const data = JSON.parse(event.data)
    console.log('收到标注更新事件:', data)
    loadList()
    loadStats()
    loadNextSample()
    if (activeTab.value === 'priority') {
      loadPriorityQueue()
    }
  } catch (error) {
    console.error('解析事件数据失败:', error)
  }
}

const initEventSource = () => {
  if (eventSource) {
    eventSource.close()
  }
  
  eventSource = new EventSource('/api/events/annotation')
  eventSource.onmessage = handleEvent
  eventSource.onerror = (error) => {
    console.error('SSE连接错误:', error)
    setTimeout(initEventSource, 5000)
  }
}

const formatTime = (isoTime) => {
  if (!isoTime) return '未更新'
  return new Date(isoTime).toLocaleString('zh-CN')
}

const getPriorityColor = (score) => {
  if (score >= 0.8) return 'red'
  if (score >= 0.5) return 'orange'
  return 'green'
}

const loadList = async () => {
  try {
    const sendParams = {
      is_annotated: filterAnnotated.value === "all" ? null : filterAnnotated.value,
      page: pagination.value.current,
      page_size: pagination.value.pageSize
    }
    const res = await axios({
      url: "/api/images",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      data: sendParams
    })
    allData.value = res.data.data
    pagination.value.total = res.data.total || 0
  } catch (err) {
    console.error("请求失败：", err)
  }
}

const loadPriorityQueue = async () => {
  try {
    const res = await axios.get('/api/annotation/priority-queue', {
      params: { limit: 50 }
    })
    priorityQueue.value = res.data.data || []
  } catch (err) {
    console.error("加载优先级队列失败：", err)
  }
}

const loadNextSample = async () => {
  try {
    const res = await axios.get('/api/annotation/next')
    nextSample.value = res.data.data
  } catch (err) {
    console.error("加载下一个样本失败：", err)
  }
}

const loadStats = async () => {
  try {
    const res = await axios.get('/api/annotation/stats')
    stats.value = res.data.data || stats.value
  } catch (err) {
    console.error("加载统计数据失败：", err)
  }
}

const goLabel = (id) => {
  window.open(`/label/${id}`, '_blank')
}

const goNextLabel = () => {
  if (nextSample.value) {
    window.open(`/label/${nextSample.value.image_id}`, '_blank')
    loadNextSample()
  }
}

const goBack = () => {
  router.push('/')
}

onMounted(() => {
  loadList()
  loadPriorityQueue()
  loadNextSample()
  loadStats()
  initEventSource()
})

onUnmounted(() => {
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
})
</script>

<style scoped>
.page-container { padding: 30px; max-width: 1400px; margin: 0 auto; }
.header { margin-bottom: 24px; display: flex; align-items: center; }
.title { margin: 0 16px 0 0; }
.desc { color: #666; margin: 0; }

.stats-panel {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
  flex-wrap: wrap;
}

.stat-card {
  flex: 1;
  min-width: 150px;
  background: #fff;
  padding: 16px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.progress-card {
  flex: 2;
  min-width: 200px;
  display: flex;
  align-items: center;
}

.quick-actions {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

:deep(.ant-tabs-tab) {
  font-size: 16px;
  font-weight: 500;
}
</style>