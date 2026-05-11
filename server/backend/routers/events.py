"""事件流路由"""
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

# 事件订阅者
subscribers = {}
lock = asyncio.Lock()

def broadcast(channel: str, message: str):
    """广播消息到指定频道的所有订阅者"""
    asyncio.create_task(_broadcast(channel, message))

async def _broadcast(channel: str, message: str):
    """异步广播消息"""
    async with lock:
        # 复制订阅者列表以避免迭代时修改
        queues = list(subscribers.get(channel, []))
        for queue in queues[:]:
            try:
                await queue.put(message)
            except asyncio.QueueEmpty:
                subscribers[channel].remove(queue)

@router.get("/api/events/{channel}")
async def event_stream(channel: str):
    """SSE事件流接口"""
    queue = asyncio.Queue(maxsize=10)
    
    async with lock:
        if channel not in subscribers:
            subscribers[channel] = []
        subscribers[channel].append(queue)
    
    async def event_generator():
        try:
            while True:
                message = await queue.get()
                yield f"data: {message}\n\n"
        except asyncio.CancelledError:
            async with lock:
                if channel in subscribers and queue in subscribers[channel]:
                    subscribers[channel].remove(queue)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
