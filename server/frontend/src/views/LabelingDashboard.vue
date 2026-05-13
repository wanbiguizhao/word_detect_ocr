<template>
  <div class="dashboard-container">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon total">📊</div>
        <div class="stat-content">
          <div class="stat-value">{{ stats.total_images }}</div>
          <div class="stat-label">总图片数</div>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon labeled">✅</div>
        <div class="stat-content">
          <div class="stat-value">{{ stats.labeled_count }}</div>
          <div class="stat-label">已标注</div>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon prelabeled">🔄</div>
        <div class="stat-content">
          <div class="stat-value">{{ stats.prelabeled_count }}</div>
          <div class="stat-label">预标注</div>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon unlabeled">⏳</div>
        <div class="stat-content">
          <div class="stat-value">{{ stats.unlabeled_count }}</div>
          <div class="stat-label">未标注</div>
        </div>
      </div>
    </div>

    <div class="progress-section">
      <div class="progress-header">
        <span class="progress-title">标注进度</span>
        <span class="progress-percent">{{ progressPercent }}%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
      </div>
    </div>

    <div class="action-buttons">
      <button class="btn btn-primary" @click="generatePrelabels">
        🤖 生成预标注
      </button>
      <button class="btn btn-secondary" @click="runClustering">
        🔍 执行聚类
      </button>
      <button class="btn btn-outline" @click="refreshData">
        🔄 刷新数据
      </button>
    </div>
    
    <div class="refresh-info">
      最后刷新: {{ lastRefreshTime || '未刷新' }}
    </div>

    <div class="section">
      <div class="section-header">
        <h2>汉字统计</h2>
        <div class="search-box">
          <input 
            type="text" 
            v-model="searchChar" 
            placeholder="搜索汉字..."
            class="search-input"
          />
        </div>
      </div>
      
      <div class="char-table-container">
        <table class="char-table">
          <thead>
            <tr>
              <th>汉字</th>
              <th>总数</th>
              <th>已确认</th>
              <th>
                <button 
                  @click="toggleSort('pending')"
                  style="background: #3b82f6; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 14px;"
                >
                  待确认 {{ sortBy === 'pending' ? (sortOrder === 'asc' ? '↑' : '↓') : '↕' }}
                </button>
              </th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr 
              v-for="item in filteredChars" 
              :key="item.char"
              :class="{ selected: selectedRow === item.char }"
              @click="selectRow(item.char)"
            >
              <td class="char-cell">{{ item.char }}</td>
              <td>{{ item.total }}</td>
              <td>{{ item.confirmed }}</td>
              <td>{{ item.pending }}</td>
              <td>
                <button 
                  class="btn btn-sm btn-outline" 
                  @click.stop="handleViewPrelabels(item.char)"
                >
                  查看预标注
                </button>
              </td>
            </tr>
          </tbody>
        </table>
        
        <div class="pagination-container">
          <div class="pagination-info">
            共 {{ pagination.total }} 条记录，显示第 {{ (pagination.currentPage - 1) * pagination.pageSize + 1 }} - {{ Math.min(pagination.currentPage * pagination.pageSize, pagination.total) }} 条
          </div>
          <div class="pagination-controls">
            <button 
              class="btn btn-sm btn-outline" 
              :disabled="pagination.currentPage === 1"
              @click="handlePageChange(pagination.currentPage - 1)"
            >
              上一页
            </button>
            <span class="pagination-numbers">
              <span 
                v-for="page in visiblePages" 
                :key="page"
                :class="['page-number', { active: page === pagination.currentPage, disabled: page === '...' }]"
                @click="page !== '...' && handlePageChange(page)"
              >
                {{ page }}
              </span>
            </span>
            <button 
              class="btn btn-sm btn-outline" 
              :disabled="pagination.currentPage >= Math.ceil(pagination.total / pagination.pageSize)"
              @click="handlePageChange(pagination.currentPage + 1)"
            >
              下一页
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue';

const stats = ref({
  total_images: 0,
  labeled_count: 0,
  prelabeled_count: 0,
  unlabeled_count: 0,
  char_stats: []
});

const searchChar = ref('');
const sortBy = ref('');
const sortOrder = ref('desc');
const selectedRow = ref('');

const pagination = ref({
  currentPage: 1,
  pageSize: 20,
  total: 0
});

const progressPercent = computed(() => {
  if (stats.value.total_images === 0) return 0;
  return Math.round((stats.value.labeled_count / stats.value.total_images) * 100);
});

const filteredChars = computed(() => {
  if (!searchChar.value) {
    return stats.value.char_stats;
  }
  return stats.value.char_stats.filter(item => item.char.includes(searchChar.value));
});

const toggleSort = (field) => {
  if (sortBy.value === field) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc';
  } else {
    sortBy.value = field;
    sortOrder.value = 'desc';
  }
  pagination.value.currentPage = 1;
  fetchCharList();
};

const selectRow = (char) => {
  selectedRow.value = char;
};

const visiblePages = computed(() => {
  const totalPages = Math.ceil(pagination.value.total / pagination.value.pageSize);
  const currentPage = pagination.value.currentPage;
  const pages = [];
  
  if (totalPages <= 7) {
    for (let i = 1; i <= totalPages; i++) {
      pages.push(i);
    }
  } else {
    if (currentPage <= 3) {
      pages.push(1, 2, 3, 4, '...', totalPages);
    } else if (currentPage >= totalPages - 2) {
      pages.push(1, '...', totalPages - 3, totalPages - 2, totalPages - 1, totalPages);
    } else {
      pages.push(1, '...', currentPage - 1, currentPage, currentPage + 1, '...', totalPages);
    }
  }
  
  return pages;
});

const handlePageChange = (page) => {
  pagination.value.currentPage = page;
  fetchCharList();
};

const handleSizeChange = (size) => {
  pagination.value.pageSize = size;
  pagination.value.currentPage = 1;
  fetchCharList();
};

const fetchStats = async () => {
  try {
    const response = await fetch('/api/labeling/stats');
    const data = await response.json();
    stats.value = data;
  } catch (error) {
    console.error('获取统计信息失败:', error);
  }
};

const fetchCharList = async () => {
  try {
    const url = new URL('/api/labeling/char-list', window.location.origin);
    url.searchParams.set('page', pagination.value.currentPage);
    url.searchParams.set('page_size', pagination.value.pageSize);
    if (sortBy.value) {
      url.searchParams.set('sort_by', sortBy.value);
      url.searchParams.set('sort_order', sortOrder.value);
    }
    const response = await fetch(url);
    const data = await response.json();
    if (data.code === 0) {
      stats.value.char_stats = data.data;
      pagination.value.total = data.total;
    }
  } catch (error) {
    console.error('获取汉字列表失败:', error);
  }
};

const generatePrelabels = async () => {
  try {
    const response = await fetch('/api/labeling/prelabels/generate', {
      method: 'POST'
    });
    const data = await response.json();
    if (data.code === 0) {
      alert('预标注生成成功！');
      fetchStats();
    } else {
      alert('生成失败: ' + data.msg);
    }
  } catch (error) {
    console.error('生成预标注失败:', error);
  }
};

const runClustering = async () => {
  try {
    const response = await fetch('/api/labeling/clusters/run', {
      method: 'POST'
    });
    const data = await response.json();
    if (data.code === 0) {
      alert('聚类完成！');
    } else {
      alert('聚类失败: ' + data.msg);
    }
  } catch (error) {
    console.error('执行聚类失败:', error);
  }
};

const viewPrelabels = (char) => {
  window.open(`/prelabel-confirm/${encodeURIComponent(char)}`, '_blank');
};

const handleViewPrelabels = (char) => {
  selectRow(char);
  viewPrelabels(char);
};

const lastRefreshTime = ref('');
const refreshData = async () => {
  await fetchStats();
  await fetchCharList();
  lastRefreshTime.value = new Date().toLocaleString('zh-CN');
};

onMounted(() => {
  refreshData();
});
</script>

<style scoped>
.dashboard-container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 24px;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

.stat-icon.total { background: #e0f2fe; }
.stat-icon.labeled { background: #dcfce7; }
.stat-icon.prelabeled { background: #fef9c3; }
.stat-icon.unlabeled { background: #fce7f3; }

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 24px;
  font-weight: 700;
  color: #1f2937;
}

.stat-label {
  font-size: 14px;
  color: #6b7280;
}

.progress-section {
  background: white;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.progress-title {
  font-weight: 600;
  color: #374151;
}

.progress-percent {
  font-size: 18px;
  font-weight: 700;
  color: #10b981;
}

.progress-bar {
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #34d399);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.action-buttons {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
}

.refresh-info {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 24px;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: #3b82f6;
  color: white;
}

.btn-primary:hover {
  background: #2563eb;
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 12px;
}

.btn-xs {
  padding: 4px 8px;
  font-size: 11px;
}

.btn-outline {
  background: transparent;
  border: 1px solid #d1d5db;
  color: #374151;
}

.btn-outline:hover {
  background: #f3f4f6;
}

.btn-success {
  background: #10b981;
  color: white;
}

.btn-warning {
  background: #f59e0b;
  color: white;
}

.btn-danger {
  background: #ef4444;
  color: white;
}

.section {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.section-header h2 {
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
}

.search-box {
  position: relative;
}

.search-input {
  padding: 8px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 14px;
  width: 200px;
}

.char-table-container {
  overflow-x: auto;
}

.char-table {
  width: 100%;
  border-collapse: collapse;
}

.char-table th,
.char-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #e5e7eb;
}

.char-table th {
  background: #f9fafb;
  font-weight: 600;
  color: #374151;
}

.char-table tbody tr {
  cursor: pointer;
  transition: background-color 0.2s;
}

.char-table tbody tr:hover {
  background-color: #f3f4f6;
}

.char-table tbody tr.selected {
  background-color: #dbeafe;
  border-left: 4px solid #3b82f6;
}

.char-cell {
  font-size: 18px;
  font-weight: 500;
}

.sort-btn {
  background: none;
  border: none;
  padding: 0;
  font-weight: 600;
  color: #374151;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
}

.sort-btn:hover {
  color: #3b82f6;
}

.sort-btn.active {
  color: #3b82f6;
}

.pagination-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0;
  border-top: 1px solid #e5e7eb;
  margin-top: 8px;
}

.pagination-info {
  font-size: 14px;
  color: #6b7280;
}

.pagination-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.pagination-numbers {
  display: flex;
  align-items: center;
  gap: 4px;
}

.page-number {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  color: #374151;
}

.page-number:hover:not(.disabled) {
  background: #f3f4f6;
}

.page-number.active {
  background: #3b82f6;
  color: white;
}

.page-number.disabled {
  cursor: not-allowed;
  color: #9ca3af;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>