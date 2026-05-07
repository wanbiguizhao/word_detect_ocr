<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack" style="margin-right: 16px;">← 返回主页</a-button>
      <h1 class="title">推荐汉字标注列表</h1>
      <p class="desc">基于迁移学习的智能标注推荐</p>
    </div>

    <div class="stats-panel">
      <a-statistic title="源数据集汉字数" :value="stats.totalChars" />
      <a-statistic title="待标注图片数" :value="stats.totalUnlabeled" />
      <a-statistic title="已标注汉字数" :value="stats.labeledChars" />
      <a-statistic title="总匹配数" :value="stats.totalMatches" />
    </div>

    <div class="search-bar">
      <a-input
        v-model="searchKeyword"
        placeholder="搜索汉字..."
        style="width: 300px;"
        @keyup.enter="loadPriorityList"
      >
        <template #prefix>
          <SearchOutlined />
        </template>
      </a-input>
      <a-button type="primary" @click="loadPriorityList">搜索</a-button>
      <a-button @click="refreshData">刷新</a-button>
    </div>

    <div class="table-container">
      <a-table
        :columns="columns"
        :data-source="priorityChars"
        :pagination="pagination"
        :row-key="record => record.char"
        @change="handleTableChange"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'char'">
            <div class="char-cell">
              <span class="char-badge" :class="getActionClass(record.action)">
                {{ record.char }}
              </span>
              <span class="action-tag" :class="getActionClass(record.action)">
                {{ record.action }}
              </span>
            </div>
          </template>
          <template v-if="column.key === 'action'">
            <a-button type="primary" size="small" @click="goToDetail(record.char)">
              去标注
            </a-button>
          </template>
          <template v-if="column.key === 'progress'">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: record.labeledPercent + '%' }"></div>
              <span class="progress-text">{{ record.labeledPercent.toFixed(1) }}%</span>
            </div>
          </template>
        </template>
      </a-table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { SearchOutlined } from '@ant-design/icons-vue'
import axios from 'axios'

const router = useRouter()

const searchKeyword = ref('')
const priorityChars = ref([])
const pagination = ref({
  current: 1,
  pageSize: 20,
  total: 0
})

const columns = [
  {
    title: '汉字',
    key: 'char',
    dataIndex: 'char',
    width: 150
  },
  {
    title: '源数据样本数',
    key: 'source_count',
    dataIndex: 'source_count',
    width: 120,
    align: 'center'
  },
  {
    title: '目标匹配数',
    key: 'target_count',
    dataIndex: 'target_count',
    width: 120,
    align: 'center'
  },
  {
    title: '已标注数',
    key: 'labeled_count',
    dataIndex: 'labeled_count',
    width: 100,
    align: 'center'
  },
  {
    title: '标注进度',
    key: 'progress',
    width: 150
  },
  {
    title: '优先级',
    key: 'priority_score',
    dataIndex: 'priority_score',
    width: 100,
    align: 'center'
  },
  {
    title: '操作',
    key: 'action',
    width: 100,
    align: 'center'
  }
]

const stats = computed(() => {
  const totalChars = priorityChars.value.length
  const totalUnlabeled = priorityChars.value.reduce((sum, item) => sum + (item.target_count - item.labeled_count), 0)
  const labeledChars = priorityChars.value.filter(item => item.labeled_count > 0).length
  const totalMatches = priorityChars.value.reduce((sum, item) => sum + item.target_count, 0)
  
  return {
    totalChars,
    totalUnlabeled,
    labeledChars,
    totalMatches
  }
})

const handleTableChange = (paginationInfo) => {
  pagination.value = paginationInfo
}

const getActionClass = (action) => {
  if (action === '优先标注') return 'priority'
  if (action === '自动迁移') return 'auto'
  return 'supplement'
}

const goBack = () => {
  router.push('/')
}

const goToDetail = (char) => {
  window.open(`/migration-detail/${char}`, '_blank')
}

const loadPriorityList = async () => {
  try {
    const params = {
      keyword: searchKeyword.value || undefined
    }
    const res = await axios.get('/api/migration/priority-list', { params })
    if (res.data.code === 0) {
      const data = res.data.data || []
      priorityChars.value = data.map(item => ({
        ...item,
        target_count: Number(item.target_count) || 0,
        labeled_count: Number(item.labeled_count) || 0,
        source_count: Number(item.source_count) || 0,
        labeledPercent: item.target_count > 0 ? (item.labeled_count / item.target_count) * 100 : 0
      }))
      pagination.value.total = data.length
    }
  } catch (err) {
    console.error('加载优先级列表失败:', err)
  }
}

const refreshData = () => {
  searchKeyword.value = ''
  loadPriorityList()
}

onMounted(() => {
  loadPriorityList()
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

.stats-panel {
  display: flex;
  gap: 24px;
  margin-bottom: 24px;
  padding: 16px;
  background: #f5f5f5;
  border-radius: 8px;
}

.search-bar {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
}

.table-container {
  background: #fff;
  border-radius: 8px;
  padding: 16px;
}

.char-cell {
  display: flex;
  align-items: center;
  gap: 8px;
}

.char-badge {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  font-weight: bold;
  border-radius: 8px;
}

.char-badge.priority {
  background: #fff1f0;
  color: #ff4d4f;
}

.char-badge.auto {
  background: #e6f7ff;
  color: #1890ff;
}

.char-badge.supplement {
  background: #f6ffed;
  color: #52c41a;
}

.action-tag {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 4px;
}

.action-tag.priority {
  background: #fff1f0;
  color: #ff4d4f;
}

.action-tag.auto {
  background: #e6f7ff;
  color: #1890ff;
}

.action-tag.supplement {
  background: #f6ffed;
  color: #52c41a;
}

.progress-bar {
  position: relative;
  height: 20px;
  background: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #52c41a, #73d13d);
  transition: width 0.3s ease;
}

.progress-text {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  text-align: center;
  font-size: 12px;
  color: #666;
  line-height: 20px;
}
</style>