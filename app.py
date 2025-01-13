from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from routes.agent_routes import router as agent_router
from routes.chat_routes import router as chat_router

app = FastAPI()

# CORS 配置
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 確保存在上傳文件目錄
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 設置靜態文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 設置模板
templates = Jinja2Templates(directory="templates")

# 註冊路由器 - 不要加前綴
app.include_router(agent_router)
app.include_router(chat_router)

@app.get("/")
async def home(request: Request):
    """首頁"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat")
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/knowledge-base")
async def knowledge_base(request: Request):
    stats = {
        "file_count": 10,
        "total_chunks": 100,
        "total_size": "1.2MB"
    }
    knowledge_base = {
        "file1.txt": ["chunk1", "chunk2"],
        "file2.txt": ["chunk3", "chunk4"]
    }
    return templates.TemplateResponse(
        "knowledge_base.html", 
        {
            "request": request, 
            "stats": stats,
            "knowledge_base": knowledge_base
        }
    )

if __name__ == "__main__":
    import uvicorn
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    uvicorn.run(
        "app:app",
        host="127.0.0.1",    # 使用 127.0.0.1 而不是 localhost
        port=8000,
        reload=True,
        ws_max_size=1024*1024,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*"
    )