<template>
  <div class="label-page">
    <!-- 头部 -->
    <div class="header">
      <a-button @click="goBack" size="large">返回列表</a-button>
      <h2>标注编辑：{{ imageId }}</h2>
      <div class="annotation-status">
        <a-tag v-if="isPostponed" color="purple" size="large">
          暂不标注
        </a-tag>
        <a-tag v-else-if="isAnnotated" color="green" size="large">
          已标注
        </a-tag>
        <a-tag v-else color="orange" size="large">
          未标注
        </a-tag>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <a-spin tip="加载中..." size="large" />
    </div>

    <!-- 内容区域 -->
    <div v-else class="content">
      <!-- 原始图片 -->
      <div class="card">
        <div class="card-header">
          <h3>原始图片</h3>
        </div>
        <div class="card-body">
          <img :src="imgUrl" class="original-image" alt="原始图片" />
        </div>
      </div>

      <!-- 规则切割 -->
      <div class="card">
        <div class="card-header">
          <h3>规则切割结果</h3>
          <a-button size="small" type="primary" @click="useRuleLines">使用规则切割</a-button>
        </div>
        <div class="card-body">
          <LineCanvas 
            :image-url="imgUrl" 
            :lines="ruleLines" 
            v-model:selected-indexes="ruleSelected"
            readonly 
            @select-change="handleSelectChange('rule')"
          />
        </div>
      </div>

      <!-- 模型切割 -->
      <div class="card">
        <div class="card-header">
          <h3>模型切割</h3>
          <a-button size="small" type="primary" @click="useModelLines">使用模型切割</a-button>
        </div>
        <div class="card-body">
          <LineCanvas 
            :image-url="imgUrl" 
            :lines="modelLines" 
            v-model:selected-indexes="modelSelected"
            readonly 
            @select-change="handleSelectChange('model')"
          />
        </div>
      </div>

      <!-- 融合切割 -->
      <div class="card">
        <div class="card-header">
          <h3>融合切割</h3>
          <a-button size="small" type="primary" @click="useFusionLines">使用融合切割</a-button>
        </div>
        <div class="card-body">
          <LineCanvas 
            :image-url="imgUrl" 
            :lines="fusionLines" 
            v-model:selected-indexes="fusionSelected"
            readonly 
            @select-change="handleSelectChange('fusion')"
          />
        </div>
      </div>

      <!-- 可编辑区域 -->
      <div class="card edit-card">
        <div class="card-header">
          <h3>✏️ 可编辑切割线</h3>
          <div class="line-count">
            线条数量: {{ editLines.length }}
          </div>
        </div>
        <div class="card-body">
          <div class="action-buttons">
            <a-space wrap>
              <!-- 单一统一复制按钮 -->
              <a-button type="primary" @click="copySelectedLine">复制选中分割线</a-button>
              <a-divider type="vertical" />
              <a-button danger @click="deleteSelectedLines">删除选中线</a-button>
              <a-button @click="clearAllLines">清空所有线</a-button>
              <a-divider type="vertical" />
              <a-button type="success" @click="save">保存标注</a-button>
              <a-button type="default" @click="postpone">暂不标注</a-button>
            </a-space>
          </div>
          <div class="operation-tips">
            <a-tooltip title="点击选中线条，Ctrl+左键拖拽框选多个线条">
              <span>操作提示: 点击选中线条，Ctrl+左键拖拽框选多个线条，左右箭头微调位置</span>
            </a-tooltip>
          </div>
          <LineCanvas
            :image-url="imgUrl"
            v-model:lines="editLines"
            v-model:selected-indexes="editSelected"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'
import LineCanvas from '../components/LineCanvas.vue'

const route = useRoute()
const router = useRouter()
const imageId = route.params.id

const imgUrl = `http://localhost:5000/api/images/${imageId}/raw`

// 加载状态
const loading = ref(true)

// 标注状态
const isAnnotated = ref(false)
const isPostponed = ref(false)

// 源线条数据
const ruleLines = ref([])
const modelLines = ref([])
const fusionLines = ref([])

// 三大预览区 选中数组
const ruleSelected = ref([])
const modelSelected = ref([])
const fusionSelected = ref([])

// 编辑区
const editLines = ref([])
const editSelected = ref([])

// ===================== 核心：选中互斥逻辑 =====================
const handleSelectChange = (type) => {
  if (type === 'rule') {
    modelSelected.value = []
    fusionSelected.value = []
  }
  if (type === 'model') {
    ruleSelected.value = []
    fusionSelected.value = []
  }
  if (type === 'fusion') {
    ruleSelected.value = []
    modelSelected.value = []
  }
}

watch(ruleSelected, (val) => {
  if (val.length > 0) handleSelectChange('rule')
},{ deep: true })

watch(modelSelected, (val) => {
  if (val.length > 0) handleSelectChange('model')
},{ deep: true })

watch(fusionSelected, (val) => {
  if (val.length > 0) handleSelectChange('fusion')
},{ deep: true })

// ===================== 修复：复制去重，杜绝重复绘制 =====================
const copySelectedLine = () => {
  let targetLines = []

  // 自动判断当前选中区域
  if (ruleSelected.value.length > 0) {
    targetLines = ruleSelected.value.map(idx => ruleLines.value[idx])
  } else if (modelSelected.value.length > 0) {
    targetLines = modelSelected.value.map(idx => modelLines.value[idx])
  } else if (fusionSelected.value.length > 0) {
    targetLines = fusionSelected.value.map(idx => fusionLines.value[idx])
  } else {
    return alert('请先在【规则/模型/融合】区域选中需要复制的分割线')
  }

  // ✅ 关键修复：获取编辑区已有的坐标，去重（相同pos不重复添加）
  const existPos = new Set(editLines.value.map(item => item.pos))
  const newLines = targetLines.filter(item => !existPos.has(item.pos))

  if (newLines.length === 0) {
    return alert('选中的分割线已存在于编辑区，无需重复复制')
  }

  // 仅追加不重复的线条
  const copy = JSON.parse(JSON.stringify(newLines))
  editLines.value = [...editLines.value, ...copy]
}

// ===================== 一键全量覆盖 =====================
const useRuleLines = () => {
  editLines.value = JSON.parse(JSON.stringify(ruleLines.value))
  editSelected.value = []
}
const useModelLines = () => {
  editLines.value = JSON.parse(JSON.stringify(modelLines.value))
  editSelected.value = []
}
const useFusionLines = () => {
  editLines.value = JSON.parse(JSON.stringify(fusionLines.value))
  editSelected.value = []
}

// ===================== 编辑区操作 =====================
const deleteSelectedLines = () => {
  if(editSelected.value.length === 0) return alert('请选中可编辑区域线条')
  editSelected.value.sort((a,b)=>b-a).forEach(i=>editLines.value.splice(i,1))
  editSelected.value = []
}
const clearAllLines = () => {
  if(confirm('确定清空所有切割线吗？')) {
    editLines.value = []
    editSelected.value = []
  }
}

// 保存
const save = async () => {
  try {
    await axios.post(`http://localhost:5000/api/images/${imageId}/annotate`, { lines: editLines.value })
    alert('保存成功！')
  } catch (error) {
    alert('保存失败，请重试')
    console.error('保存失败:', error)
  }
}

// 暂不标注
const postpone = async () => {
  try {
    await axios.post(`http://localhost:5000/api/images/${imageId}/postpone`)
    alert('已标记为暂不标注！')
  } catch (error) {
    alert('操作失败，请重试')
    console.error('暂不标注失败:', error)
  }
}

// 加载数据
const loadDetail = async () => {
  try {
    loading.value = true
    const res = await axios.get(`http://localhost:5000/api/images/${imageId}/detail`)
    const data = res.data.data
    isAnnotated.value = data.is_annotated || false
    isPostponed.value = data.is_postponed || false
    ruleLines.value = data.rule_lines || []
    modelLines.value = data.model_lines || []
    fusionLines.value = data.fusion_lines || []
    editLines.value = data.annotation?.lines || []
  } catch (error) {
    alert('加载数据失败，请重试')
    console.error('加载数据失败:', error)
  } finally {
    loading.value = false
  }
}

const goBack = () => router.back()
onMounted(() => loadDetail())
</script>

<style scoped>
.label-page {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
  box-sizing: border-box;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 30px;
  padding-bottom: 15px;
  border-bottom: 1px solid #e8e8e8;
}

.header h2 {
  margin: 0;
  font-size: 28px;
  font-weight: 600;
  color: #333;
}

.annotation-status {
  margin-left: auto;
}

.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 600px;
}

.content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.card {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
  overflow: hidden;
  transition: all 0.3s ease;
}

.card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background: #fafafa;
  border-bottom: 1px solid #e8e8e8;
}

.card-header h3 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: #333;
}

.line-count {
  font-size: 16px;
  color: #666;
  background: #f0f0f0;
  padding: 4px 12px;
  border-radius: 12px;
}

.card-body {
  padding: 20px;
  overflow-x: auto;
}

.original-image {
  display: block;
  border: 1px solid #e8e8e8;
  border-radius: 4px;
  object-fit: contain;
  max-width: 100%;
}

.edit-card {
  margin-top: 20px;
}

/* 确保LineCanvas容器不缩小图像 */
:deep(canvas) {
  max-width: 100%;
  height: auto !important;
}

.action-buttons {
  margin-bottom: 15px;
  text-align: center;
}

.operation-tips {
  margin-bottom: 15px;
  padding: 10px;
  background: #f6ffed;
  border: 1px solid #b7eb8f;
  border-radius: 4px;
  font-size: 14px;
  color: #389e0d;
  text-align: center;
}

@media (max-width: 768px) {
  .label-page {
    padding: 10px;
  }
  
  .header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  
  .cutting-results {
    grid-template-columns: 1fr;
  }
  
  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  
  .action-buttons {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
  }
}
</style>