<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack">← 返回列表</a-button>
      <h2 class="title">聚类 {{ clusterId }} 修改标注</h2>
      <a-button type="primary" @click="saveChanges" :loading="saving">保存修改</a-button>
    </div>

    <div class="image-grid">
      <div
        v-for="img in images"
        :key="img.index"
        :class="['image-card', getStatusClass(img)]"
        :data-index="img.index"
      >
        <img :src="img.path" :alt="img.label || '未标注'" />
        <div class="label-area">
          <a-input
            :value="img.label"
            placeholder="标注"
            style="width: 60px;"
            @input="updateLabel(img.index, $event)"
            @click.stop
          />
        </div>
        <span class="image-info">{{ img.info }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const clusterId = ref(route.params.id)
const images = ref([])
const saving = ref(false)
const originalLabels = ref({})
const changes = ref({})

const getStatusClass = (img) => {
  const hasChange = changes.value[img.index] !== undefined
  const newValue = changes.value[img.index]
  const oldValue = originalLabels.value[img.index]
  
  if (hasChange) {
    if (newValue) {
      return 'editing'
    } else {
      return 'unlabeled'
    }
  }
  return img.label ? 'labeled' : 'unlabeled'
}

const updateLabel = (index, event) => {
  const value = event.target.value
  if (value) {
    changes.value[index] = value
  } else {
    delete changes.value[index]
  }
}

const loadImages = async () => {
  try {
    const response = await axios.get(`/api/clusters/${clusterId.value}/images`)
    images.value = response.data.images
    images.value.forEach(img => {
      originalLabels.value[img.index] = img.label
    })
  } catch (err) {
    console.error('加载图片失败:', err)
  }
}

const saveChanges = async () => {
  saving.value = true
  
  try {
    const toSave = []
    for (const [index, char] of Object.entries(changes.value)) {
      toSave.push({ charIndex: parseInt(index), char })
    }
    
    if (toSave.length > 0) {
      await axios.post('/api/cluster-labels/batch-save', {
        clusterId: parseInt(clusterId.value),
        labels: toSave
      })
      
      alert('修改成功！')
      router.push('/ocr')
    } else {
      alert('没有修改内容')
    }
  } catch (err) {
    console.error('保存失败:', err)
    alert('保存失败')
  } finally {
    saving.value = false
  }
}

const goBack = () => {
  router.push('/ocr')
}

onMounted(() => {
  loadImages()
})
</script>

<style scoped>
.page-container {
  padding: 20px;
}

.header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.title {
  flex: 1;
  margin: 0;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 12px;
}

.image-card {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  padding: 8px;
  transition: all 0.3s ease;
}

.image-card.unlabeled {
  border: 2px solid #e8e8e8;
  background: #fff;
}

.image-card.editing {
  border: 2px dashed #faad14;
  background: #fffbe6;
  animation: pending-pulse 2s infinite;
}

.image-card.labeled {
  border: 2px solid #52c41a;
  background: #f6ffed;
}

.image-card img {
  width: 100%;
  height: 80px;
  object-fit: contain;
  background: #f5f5f5;
  border-radius: 4px;
}

.label-area {
  margin-top: 8px;
  text-align: center;
}

.image-info {
  display: block;
  font-size: 10px;
  color: #999;
  margin-top: 4px;
  text-align: center;
}

@keyframes pending-pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(250, 173, 20, 0.4);
  }
  50% {
    box-shadow: 0 0 0 6px rgba(250, 173, 20, 0);
  }
}
</style>