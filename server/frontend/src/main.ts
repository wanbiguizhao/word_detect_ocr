import { createApp } from 'vue'
// @ts-ignore
import App from './App.vue'
// @ts-ignore
import router from './router/index'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'

const app = createApp(App)
app.use(router)
app.use(Antd)
app.mount('#app')