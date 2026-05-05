from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="主动学习汉字分割标注系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.middleware("http")
async def set_encoding(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


from routers.images import router as images_router
from routers.clusters import router as clusters_router
from routers.pseudo_labels import router as pseudo_labels_router
from routers.line_status import router as line_status_router

app.include_router(line_status_router)
app.include_router(images_router)
app.include_router(clusters_router)
app.include_router(pseudo_labels_router)