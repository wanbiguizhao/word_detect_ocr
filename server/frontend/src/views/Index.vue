<template>
  <div class="page-container">
    <div class="header">
      <h1 class="title">汉字切割标注系统</h1>
      <p class="desc">主动学习样本 | 数据来源：top_annotate_samples.json</p>
    </div>

    <!-- 移除v-model，纯靠change事件强制赋值 -->
    <div style="margin-bottom: 16px;">
      <a-select
        placeholder="筛选标注状态"
        style="width: 160px"
        @change="handleSelectChange"
      >
        <a-select-option value="all">全部样本</a-select-option>
        <a-select-option value="true">已标注</a-select-option>
        <a-select-option value="false">未标注</a-select-option>
      </a-select>
    </div>

    <a-table
      :data-source="list"
      row-key="id"
      bordered
      :pagination="{ pageSize: 10 }"
    >
      <a-table-column title="文件名称" data-index="image_name" width="300" />
      <a-table-column title="主动学习分数" width="150">
        <template #default="{ record }">
          {{ record.score.toFixed(4) }}
        </template>
      </a-table-column>
      <a-table-column title="标注状态" width="130">
        <template #default="{ record }">
          <a-tag :color="record.is_annotated ? 'green' : 'orange'">
            {{ record.is_annotated ? '已标注' : '未标注' }}
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
          <a-button type="primary" @click="goLabel(record.id)">进入标注</a-button>
        </template>
      </a-table-column>
    </a-table>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const list = ref([])
// 初始值
const filterAnnotated = ref("all")

// ==============================================
// 🔥 核心修复：手动强制赋值！彻底解决绑定失效！
// ==============================================
const handleSelectChange = (selectedValue) => {
  // 强制把选中的值赋值给变量
  filterAnnotated.value = selectedValue
  
  console.log("==================================================")
  console.log("用户选中：", selectedValue)
  console.log("变量已强制更新为：", filterAnnotated.value)
  console.log("==================================================")
  
  loadList()
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

    console.log("📤 最终发送参数：", sendParams)

    const res = await axios({
      url: "http://localhost:5000/api/images",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      data: sendParams
    })

    list.value = res.data.data

  } catch (err) {
    console.error("请求失败：", err)
  }
}

const goLabel = (id) => {
  router.push(`/label/${id}`)
}

onMounted(() => {
  loadList()
})
</script>

<style scoped>
.page-container { padding: 30px; max-width: 1400px; margin: 0 auto; }
.header { margin-bottom: 24px; }
</style>