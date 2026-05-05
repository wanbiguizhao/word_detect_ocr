import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '../views/HomePage.vue'
import SegmentLabelPage from '../views/SegmentLabelPage.vue'
import LabelPage from '../views/LabelPage.vue'
import OcrLabelPage from '../views/OcrLabelPage.vue'
import OcrClusterLabelPage from '../views/OcrClusterLabelPage.vue'
import OcrClusterEditPage from '../views/OcrClusterEditPage.vue'
import PseudoLabelListPage from '../views/PseudoLabelListPage.vue'
import PseudoLabelDetailPage from '../views/PseudoLabelDetailPage.vue'
import CharManagementPage from '../views/CharManagementPage.vue'
import LineCheckPage from '../views/LineCheckPage.vue'
import LineDetailPage from '../views/LineDetailPage.vue'

const routes = [
  { path: '/', name: 'home', component: HomePage },
  { path: '/segment', name: 'segment', component: SegmentLabelPage },
  { path: '/label/:id', name: 'label', component: LabelPage },
  { path: '/ocr', name: 'ocr', component: OcrLabelPage },
  { path: '/ocr-label/:id', name: 'ocr-label', component: OcrClusterLabelPage },
  { path: '/ocr-edit/:id', name: 'ocr-edit', component: OcrClusterEditPage },
  { path: '/pseudo-label', name: 'pseudo-label', component: PseudoLabelListPage },
  { path: '/pseudo-label/:char', name: 'pseudo-label-detail', component: PseudoLabelDetailPage },
  { path: '/char-management', name: 'char-management', component: CharManagementPage },
  { path: '/line-check', name: 'line-check', component: LineCheckPage },
  { path: '/line-detail/:lineName', name: 'line-detail', component: LineDetailPage },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router