import { createRouter, createWebHistory } from 'vue-router'
import Index from '../views/Index.vue'
import LabelPage from '../views/LabelPage.vue'

const routes = [
  { path: '/', name: 'index', component: Index },
  { path: '/label/:id', name: 'label', component: LabelPage },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router