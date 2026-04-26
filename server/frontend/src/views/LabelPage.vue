<template>
  <div style="padding: 10px; width: 100%; box-sizing: border-box;">
    <a-button @click="goBack" size="large">返回列表</a-button>
    <h2 style="margin:10px 0">标注编辑：{{ imageId }}</h2>

    <!-- 四行一列 垂直布局 -->
    <div style="display: flex; flex-direction: column; gap: 12px; margin: 20px 0;">
      <!-- 原始图片 -->
      <div>
        <h3 style="margin:0 0 8px; text-align: center;">原始图片</h3>
        <img 
          :src="imgUrl" 
          style="width:100%; display:block; border:1px solid #ccc;" 
          alt="原始图片"
        />
      </div>

      <!-- 规则切割：标题+按钮 水平居中 -->
      <div>
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
          <h3 style="margin:0;">规则切割结果</h3>
          <a-button 
            size="small" 
            type="primary" 
            @click="useRuleLines" 
            style="margin-left: 12px;"
          >使用规则切割</a-button>
        </div>
        <LineCanvas 
          :image-url="imgUrl" 
          :lines="ruleLines" 
          v-model:selected-indexes="ruleSelected"
          readonly 
          @select-change="handleSelectChange('rule')"
        />
      </div>

      <!-- 模型切割：标题+按钮 水平居中 -->
      <div>
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
          <h3 style="margin:0;">模型切割</h3>
          <a-button 
            size="small" 
            type="primary" 
            @click="useModelLines" 
            style="margin-left: 12px;"
          >使用模型切割</a-button>
        </div>
        <LineCanvas 
          :image-url="imgUrl" 
          :lines="modelLines" 
          v-model:selected-indexes="modelSelected"
          readonly 
          @select-change="handleSelectChange('model')"
        />
      </div>

      <!-- 融合切割：标题+按钮 水平居中 -->
      <div>
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
          <h3 style="margin:0;">融合切割</h3>
          <a-button 
            size="small" 
            type="primary" 
            @click="useFusionLines" 
            style="margin-left: 12px;"
          >使用融合切割</a-button>
        </div>
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
    <div style="margin-top: 30px; width:100%;">
      <h3 style="margin:0 0 12px; text-align: center;">✏️ 可编辑切割线</h3>

      <div style="margin:0 0 16px; text-align: center;">
        <a-space wrap>
          <!-- 单一统一复制按钮 -->
          <a-button type="primary" @click="copySelectedLine">复制选中分割线</a-button>

          <a-divider type="vertical" />

          <a-button danger @click="deleteSelectedLines">删除选中线</a-button>
          <a-button @click="clearAllLines">清空所有线</a-button>
          <a-button type="success" @click="save">保存标注</a-button>
        </a-space>
      </div>

      <LineCanvas
        :image-url="imgUrl"
        v-model:lines="editLines"
        v-model:selected-indexes="editSelected"
      />
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
  await axios.post(`http://localhost:5000/api/images/${imageId}/annotate`, { lines: editLines.value })
  alert('保存成功！')
}

// 加载数据
const loadDetail = async () => {
  const res = await axios.get(`http://localhost:5000/api/images/${imageId}/detail`)
  const data = res.data.data
  ruleLines.value = data.rule_lines || []
  modelLines.value = data.model_lines || []
  fusionLines.value = data.fusion_lines || []
  editLines.value = data.annotation?.lines || []
}

const goBack = () => router.back()
onMounted(() => loadDetail())
</script>