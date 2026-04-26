<template>
  <div class="page-container">
    <div class="header">
      <h1 class="title">汉字切割标注系统</h1>
      <p class="desc">主动学习样本 | 数据来源：top_annotate_samples.json</p>
    </div>

    <!-- 筛选下拉框 -->
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

    <!-- 🔥 前端分页实现 -->
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
          <!-- 🔥 点击新开标签页 -->
          <a-button type="primary" @click="goLabel(record.id)">
            进入标注
          </a-button>
        </template>
      </a-table-column>
    </a-table>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const allData = ref([])
const filterAnnotated = ref("all")

// 🔥 响应式分页配置（前端分页）
const pagination = ref({
  pageSize: 10,
  current: 1,
  showSizeChanger: true,
  pageSizeOptions: [10, 20, 50, 100],
  total: 0
})

// 计算当前页显示的数据
const paginatedList = computed(() => {
  const start = (pagination.value.current - 1) * pagination.value.pageSize
  const end = start + pagination.value.pageSize
  return allData.value.slice(start, end)
})

// 筛选切换事件
const handleSelectChange = (selectedValue) => {
  filterAnnotated.value = selectedValue
  pagination.value.current = 1 // 筛选重置为第一页
  loadList()
}

// 分页切换事件
const handlePageChange = (page) => {
  console.log('分页切换事件:', page)
  pagination.value.current = page.current
  pagination.value.pageSize = page.pageSize
  console.log('更新后分页配置:', pagination.value)
  console.log('计算后当前页数据:', paginatedList.value)
  // 前端分页，不需要重新请求数据
}

// 时间格式化
const formatTime = (isoTime) => {
  if (!isoTime) return '未更新'
  return new Date(isoTime).toLocaleString('zh-CN')
}

// 请求列表数据（一次性获取所有数据）
const loadList = async () => {
  try {
    console.log('开始加载数据，筛选条件:', filterAnnotated.value)
    const sendParams = {
      is_annotated: filterAnnotated.value === "all" ? null : filterAnnotated.value
      // 移除分页参数，一次性获取所有数据
    }
    console.log('发送参数:', sendParams)
    const res = await axios({
      url: "http://localhost:5000/api/images",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      data: sendParams
    })
    console.log('响应数据:', res.data)
    allData.value = res.data.data
    console.log('加载数据长度:', allData.value.length)
    pagination.value.total = res.data.total || allData.value.length
    console.log('更新分页total:', pagination.value.total)
    console.log('当前分页配置:', pagination.value)
  } catch (err) {
    console.error("请求失败：", err)
  }
}

// ==============================================
// 🔥 核心：新开标签页打开标注页（原页面状态永不丢失）
// ==============================================
const goLabel = (id) => {
  window.open(`/label/${id}`, '_blank')
}

onMounted(() => {
  loadList()
})
</script>

<style scoped>
.page-container { padding: 30px; max-width: 1400px; margin: 0 auto; }
.header { margin-bottom: 24px; }
</style>