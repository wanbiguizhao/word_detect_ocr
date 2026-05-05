<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack">← 返回</a-button>
      <h2 class="title">数据统计与汉字管理</h2>
      <div class="search-box">
        <a-input 
          v-model:value="searchKeyword" 
          placeholder="搜索汉字" 
          style="width: 200px;"
          @input="handleSearch"
          @keyup.enter="handleSearch"
        />
        <a-button @click="handleSearch" type="primary">搜索</a-button>
      </div>
    </div>

    <div class="stats-section">
      <div class="stats-row">
        <a-statistic title="已标注汉字" :value="totalChars" />
        <a-statistic title="涉及聚类" :value="totalClusters" />
        <a-statistic title="标注图片" :value="totalImages" />
        <a-statistic title="平均每字图片数" :value="avgImagesPerChar" />
      </div>
      
      <div class="stats-chart">
        <div class="chart-title">标注分布（Top 10）</div>
        <div class="chart-bars">
          <div 
            v-for="item in topChars" 
            :key="item.char"
            class="chart-bar-item"
          >
            <span class="bar-label">{{ item.char }}</span>
            <div class="bar-container">
              <div 
                class="bar-fill" 
                :style="{ width: (item.total_count / maxCharCount * 100) + '%' }"
              ></div>
            </div>
            <span class="bar-value">{{ item.total_count }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="char-grid">
      <div 
        v-for="item in filteredChars" 
        :key="item.char"
        class="char-card"
        @click="selectChar(item.char)"
      >
        <div class="char-display">{{ item.char }}</div>
        <div class="char-stats">
          <span>图片: {{ item.total_count }}</span>
          <span>聚类: {{ item.cluster_count }}</span>
        </div>
      </div>
    </div>

    <a-modal v-model:open="showDetailModal" :title="`汉字「${selectedChar}」详情`" :footer="null" width="900px">
      <div v-if="selectedCharData" class="detail-content">
        <div class="detail-header">
          <span class="selected-char">{{ selectedChar }}</span>
          <span class="char-count">共 {{ selectedCharData.total_count }} 张图片，涉及 {{ selectedCharData.cluster_count }} 个聚类</span>
        </div>
        
        <div class="tabs-container">
          <a-tabs v-model:activeKey="activeTab" @change="handleTabChange">
            <a-tab-pane key="images" tab="图片列表">
              <div class="images-section">
                <div v-if="charImages.length > 0">
                  <div class="images-grid">
                    <div 
                      v-for="(img, index) in charImages" 
                      :key="index"
                      class="image-item"
                      @click="goToCluster(img.cluster_id)"
                    >
                      <img :src="`http://localhost:5000/api/char-images/${img.filename}`" :alt="selectedChar" />
                      <span class="image-info">{{ img.filename }}</span>
                      <span class="cluster-tag">聚类 {{ img.cluster_id }}</span>
                    </div>
                  </div>
                </div>
                <div v-else class="empty-state">
                  暂无图片
                </div>
              </div>
            </a-tab-pane>
            
            <a-tab-pane key="labeled" tab="已标注聚类">
              <div class="clusters-section">
                <div v-if="selectedCharData.clusters && selectedCharData.clusters.length > 0">
                  <div class="cluster-list">
                    <div 
                      v-for="clusterId in selectedCharData.clusters" 
                      :key="clusterId"
                      class="cluster-item"
                    >
                      <a-button @click="goToCluster(clusterId)" type="link">
                        聚类 {{ clusterId }}
                      </a-button>
                    </div>
                  </div>
                </div>
                <div v-else class="empty-state">
                  暂无已标注聚类
                </div>
              </div>
            </a-tab-pane>
            
            <a-tab-pane key="recommend" tab="伪标签推荐">
              <div class="recommend-section">
                <div v-if="!showRecommendResult">
                  <div class="recommend-intro">
                    <p>伪标签传播可以基于已标注的「{{ selectedChar }}」字，在未标注的聚类中查找相似的图片。</p>
                    <p>系统会计算每个未标注聚类与已标注锚点的相似度，推荐相似度高的聚类供您审核。</p>
                  </div>
                  <a-button 
                    type="primary" 
                    size="large" 
                    :loading="isComputing"
                    @click="computePseudoLabels"
                  >
                    {{ isComputing ? '正在计算...' : '开始伪标签传播' }}
                  </a-button>
                </div>
                
                <div v-else>
                  <div v-if="recommendClusters.length > 0">
                    <div class="recommend-stats">
                      <span>共找到 {{ recommendClusters.length }} 个推荐聚类</span>
                      <a-button size="small" @click="refreshRecommend">刷新</a-button>
                    </div>
                    <a-table
                      :data-source="recommendClusters"
                      row-key="cluster_id"
                      bordered
                      :pagination="{ pageSize: 10 }"
                    >
                      <a-table-column title="聚类ID" data-index="cluster_id" />
                      <a-table-column title="匹配数量" data-index="matched_count" />
                      <a-table-column title="平均相似度" width="120">
                        <template #default="{ record }">
                          <a-tag :color="getSimilarityColor(record.avg_similarity)">
                            {{ (record.avg_similarity * 100).toFixed(1) }}%
                          </a-tag>
                        </template>
                      </a-table-column>
                      <a-table-column title="总图片数" data-index="total_count" />
                      <a-table-column title="操作" width="150">
                        <template #default="{ record }">
                          <a-button type="primary" size="small" @click="goToCluster(record.cluster_id)">
                            查看聚类
                          </a-button>
                        </template>
                      </a-table-column>
                    </a-table>
                  </div>
                  <div v-else class="empty-state">
                    <p>未找到推荐聚类</p>
                    <p style="font-size: 12px; color: #999;">可能是没有足够的已标注锚点，或者没有未标注的聚类</p>
                  </div>
                </div>
              </div>
            </a-tab-pane>
          </a-tabs>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()

const chars = ref([])
const searchKeyword = ref('')
const showDetailModal = ref(false)
const selectedChar = ref('')
const selectedCharData = ref(null)
const activeTab = ref('images')

const isComputing = ref(false)
const showRecommendResult = ref(false)
const recommendClusters = ref([])
const charImages = ref([])

const totalChars = computed(() => chars.value.length)
const totalClusters = computed(() => {
  const clusters = new Set()
  chars.value.forEach(item => {
    item.clusters.forEach(cid => clusters.add(cid))
  })
  return clusters.size
})
const totalImages = computed(() => {
  return chars.value.reduce((sum, item) => sum + item.total_count, 0)
})
const avgImagesPerChar = computed(() => {
  if (totalChars.value === 0) return 0
  return Math.round(totalImages.value / totalChars.value)
})
const topChars = computed(() => {
  return [...chars.value].sort((a, b) => b.total_count - a.total_count).slice(0, 10)
})
const maxCharCount = computed(() => {
  if (chars.value.length === 0) return 1
  return Math.max(...chars.value.map(item => item.total_count))
})

const filteredChars = computed(() => {
  if (!searchKeyword.value) {
    return chars.value
  }
  return chars.value.filter(item => item.char.includes(searchKeyword.value))
})

const loadData = async () => {
  try {
    const res = await axios.get('http://localhost:5000/api/pseudo-labels')
    if (res.data.code === 0) {
      chars.value = res.data.data
    }
  } catch (err) {
    console.error('加载数据失败:', err)
  }
}

const handleSearch = () => {
}

const selectChar = async (char) => {
  selectedChar.value = char
  selectedCharData.value = chars.value.find(item => item.char === char)
  activeTab.value = 'images'
  showRecommendResult.value = false
  recommendClusters.value = []
  
  try {
    const res = await axios.get(`http://localhost:5000/api/char-images/list?char=${encodeURIComponent(char)}`)
    if (res.data.code === 0) {
      charImages.value = res.data.images || []
    } else {
      charImages.value = []
    }
  } catch (err) {
    console.error('获取图片列表失败:', err)
    charImages.value = []
  }
  
  showDetailModal.value = true
}

const handleTabChange = (key) => {}

const computePseudoLabels = async () => {
  isComputing.value = true
  showRecommendResult.value = false
  
  try {
    const res = await axios.post('http://localhost:5000/api/pseudo-labels/clusters', {
      char: selectedChar.value
    })
    
    if (res.data.code === 0) {
      recommendClusters.value = res.data.clusters || []
      showRecommendResult.value = true
    }
  } catch (err) {
    console.error('伪标签传播失败:', err)
    alert('伪标签传播失败')
  } finally {
    isComputing.value = false
  }
}

const refreshRecommend = () => {
  showRecommendResult.value = false
  recommendClusters.value = []
}

const getSimilarityColor = (similarity) => {
  if (similarity >= 0.9) return 'green'
  if (similarity >= 0.8) return 'cyan'
  if (similarity >= 0.75) return 'blue'
  return 'orange'
}

const goToCluster = (clusterId) => {
  showDetailModal.value = false
  router.push(`/ocr-label/${clusterId}`)
}

const goBack = () => {
  router.push('/')
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.page-container { padding: 20px; max-width: 1400px; margin: 0 auto; }
.header { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
.title { margin: 0; }
.search-box { margin-left: auto; display: flex; gap: 8px; z-index: 10; position: relative; }

.stats-section {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.stats-row {
  display: flex;
  gap: 24px;
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid #e8e8e8;
}

.stats-chart {
  padding: 16px;
  background: #fafafa;
  border-radius: 8px;
}

.chart-title {
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 12px;
  color: #666;
}

.chart-bars {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chart-bar-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.bar-label {
  width: 32px;
  font-size: 14px;
  font-weight: 500;
}

.bar-container {
  flex: 1;
  height: 24px;
  background: #e8e8e8;
  border-radius: 4px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #1890ff, #69c0ff);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.bar-value {
  width: 48px;
  text-align: right;
  font-size: 12px;
  color: #999;
}

.char-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 12px;
}

.char-card {
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}

.char-card:hover {
  border-color: #1890ff;
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.2);
}

.char-display {
  font-size: 32px;
  font-weight: bold;
  color: #1890ff;
  margin-bottom: 8px;
}

.char-stats {
  display: flex;
  justify-content: center;
  gap: 12px;
  font-size: 12px;
  color: #999;
}

.detail-content {
  padding: 16px;
}

.detail-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid #e8e8e8;
}

.selected-char {
  font-size: 48px;
  font-weight: bold;
  color: #1890ff;
}

.char-count {
  font-size: 14px;
  color: #666;
}

.tabs-container {
  margin-top: 16px;
}

.clusters-section {
  padding: 16px;
}

.cluster-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.cluster-item {
  padding: 4px 12px;
  background: #f5f5f5;
  border-radius: 4px;
}

.images-section {
  padding: 16px;
}

.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
}

.image-item {
  text-align: center;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 8px;
  background: #fafafa;
  cursor: pointer;
  transition: all 0.2s;
}

.image-item:hover {
  border-color: #1890ff;
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.2);
}

.image-item img {
  width: 100%;
  max-height: 120px;
  object-fit: contain;
  border-radius: 4px;
}

.image-info {
  display: block;
  margin-top: 8px;
  font-size: 11px;
  color: #999;
  word-break: break-all;
}

.cluster-tag {
  display: inline-block;
  margin-top: 4px;
  padding: 2px 8px;
  font-size: 11px;
  background: #1890ff;
  color: white;
  border-radius: 4px;
}

.recommend-section {
  padding: 16px;
}

.recommend-intro {
  margin-bottom: 20px;
  padding: 16px;
  background: #f5f5f5;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.8;
}

.recommend-intro p {
  margin: 0 0 8px 0;
}

.recommend-intro p:last-child {
  margin-bottom: 0;
}

.recommend-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  font-size: 14px;
}

.empty-state {
  padding: 40px;
  text-align: center;
  background: #fafafa;
  border-radius: 8px;
  color: #999;
}
</style>