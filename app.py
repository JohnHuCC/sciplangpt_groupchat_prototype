from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
import os
from routes.agent_routes import router as agent_router
from routes.chat_routes import router as chat_router

app = FastAPI()

# 確保存在上傳文件目錄
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 設置靜態文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 設置模板
templates = Jinja2Templates(directory="templates")

# 註冊路由器（FastAPI 中使用 include_router 替代 Flask 的 register_blueprint）
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
    # 假設你有這些數據
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
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        loop="asyncio",
        workers=1
    )