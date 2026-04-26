<template>
  <canvas 
    ref="cv" 
    style="
      width: 100%; 
      border:1px solid #ccc;
      cursor: crosshair;
      background:#fff;
      display:block;
    "
  ></canvas>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'

const props = defineProps({
  imageUrl: String,
  lines: Array,
  selectedIndexes: Array,
  readonly: { type: Boolean, default: false }
})

const emit = defineEmits(['update:lines', 'update:selectedIndexes'])
const cv = ref(null)
let img = new Image()
img.crossOrigin = 'anonymous'

// 框选多选核心变量
const isSelecting = ref(false)
const startX = ref(0)
const endX = ref(0)

// 绘制函数：普通1px / 选中柠檬黄加粗
function draw() {
  const canvas = cv.value
  if (!canvas || !img.complete || !img.width) return
  const ctx = canvas.getContext('2d')

  canvas.width = img.width
  canvas.height = img.height

  ctx.clearRect(0,0,canvas.width,canvas.height)
  ctx.drawImage(img, 0, 0)

  if(props.lines){
    props.lines.forEach((line, idx)=>{
      const isSelected = props.selectedIndexes?.includes(idx)
      if (isSelected) {
        ctx.strokeStyle = "#FFEC00"
        ctx.lineWidth = 2
      } else {
        ctx.strokeStyle = line.color
        ctx.lineWidth = 1
      }
      
      ctx.beginPath()
      ctx.moveTo(line.pos, 0)
      ctx.lineTo(line.pos, canvas.height)
      ctx.stroke()
    })
  }

  // 框选半透明蓝选区
  if (isSelecting.value) {
    const minX = Math.min(startX.value, endX.value)
    const maxX = Math.max(startX.value, endX.value)
    ctx.fillStyle = 'rgba(24, 144, 255, 0.2)'
    ctx.fillRect(minX, 0, maxX - minX, canvas.height)
    ctx.strokeStyle = '#1890ff'
    ctx.strokeRect(minX, 0, maxX - minX, canvas.height)
  }
}

// Ctrl + 左键水平拖拽框选
function onMouseDown(e) {
  if (e.ctrlKey && e.button === 0) {
    isSelecting.value = true
    const rect = cv.value.getBoundingClientRect()
    const scale = img.width / rect.width
    startX.value = (e.clientX - rect.left) * scale
    endX.value = startX.value
    draw()
  }
}

function onMouseMove(e) {
  if (!isSelecting.value) return
  const rect = cv.value.getBoundingClientRect()
  const scale = img.width / rect.width
  endX.value = (e.clientX - rect.left) * scale
  draw()
}

function onMouseUp() {
  if (!isSelecting.value) return
  isSelecting.value = false

  const minX = Math.min(startX.value, endX.value)
  const maxX = Math.max(startX.value, endX.value)

  const selected = []
  props.lines.forEach((line, idx) => {
    if (line.pos >= minX && line.pos <= maxX) {
      selected.push(idx)
    }
  })

  emit('update:selectedIndexes', selected)
  draw()
}

// 单击单选
function onCanvasClick(e) {
  if (isSelecting.value) return
  if (e.ctrlKey) return

  const rect = cv.value.getBoundingClientRect()
  const scale = img.width / rect.width
  const x = (e.clientX - rect.left) * scale
  const idx = props.lines.findIndex(line => Math.abs(line.pos - x) < 8)
  
  emit('update:selectedIndexes', idx === -1 ? [] : [idx])
}

// 键盘事件：只读画布禁止方向键/Delete
function onKeyDown(e) {
  // 只读区域直接拦截键盘修改
  if (props.readonly) return
  
  if (props.selectedIndexes?.length === 0) return

  // Delete 删除
  if (e.key === 'Delete') {
    const newLines = [...props.lines]
    props.selectedIndexes.sort((a, b) => b - a).forEach(i => newLines.splice(i, 1))
    emit('update:lines', newLines)
    emit('update:selectedIndexes', [])
    draw()
    return
  }

  // 左右箭头微调
  if (e.key === 'ArrowLeft') {
    const newLines = JSON.parse(JSON.stringify(props.lines))
    props.selectedIndexes.forEach(index => {
      newLines[index].pos -= 1
    })
    emit('update:lines', newLines)
    draw()
  }
  if (e.key === 'ArrowRight') {
    const newLines = JSON.parse(JSON.stringify(props.lines))
    props.selectedIndexes.forEach(index => {
      newLines[index].pos += 1
    })
    emit('update:lines', newLines)
    draw()
  }
}

watch(() => props.imageUrl, (url) => { img.src = url; img.onload = draw }, { immediate: true })
watch(() => props.lines, draw, { deep: true })
watch(() => props.selectedIndexes, draw, { deep: true })

onMounted(() => {
  const canvas = cv.value
  canvas.addEventListener('mousedown', onMouseDown)
  canvas.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
  canvas.addEventListener('click', onCanvasClick)
  window.addEventListener('keydown', onKeyDown)
  draw()
})
</script>