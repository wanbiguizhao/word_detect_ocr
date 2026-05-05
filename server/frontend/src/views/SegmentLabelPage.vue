<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack" style="margin-right: 16px;">← 返回主页</a-button>
      <h1 class="title">字符切割标注</h1>
      <p class="desc">主动学习样本 | 数据来源：top_annotate_samples.json</p>
    </div>

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
  </div>
</template>

<script setup>
import { ref, onMounted, computed, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const allData = ref([])
const filterAnnotated = ref("all")

const pagination = ref({
  pageSize: 10,
  current: 1,
  showSizeChanger: true,
  pageSizeOptions: [10, 20, 50, 100],
  total: 0
})

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

const handleStorageChange = (e) => {
  if (e.key === 'annotationUpdated') {
    loadList()
  }
}

const formatTime = (isoTime) => {
  if (!isoTime) return '未更新'
  return new Date(isoTime).toLocaleString('zh-CN')
}

const loadList = async () => {
  try {
    const sendParams = {
      is_annotated: filterAnnotated.value === "all" ? null : filterAnnotated.value
    }
    const res = await axios({
      url: "http://localhost:5000/api/images",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      data: sendParams
    })
    allData.value = res.data.data
    pagination.value.total = res.data.total || allData.value.length
  } catch (err) {
    console.error("请求失败：", err)
  }
}

const goLabel = (id) => {
  window.open(`/label/${id}`, '_blank')
}

const goBack = () => {
  router.push('/')
}

onMounted(() => {
  loadList()
  window.addEventListener('storage', handleStorageChange)
})

onUnmounted(() => {
  window.removeEventListener('storage', handleStorageChange)
})
</script>

<style scoped>
.page-container { padding: 30px; max-width: 1400px; margin: 0 auto; }
.header { margin-bottom: 24px; display: flex; align-items: center; }
.title { margin: 0 16px 0 0; }
.desc { color: #666; margin: 0; }
</style>
