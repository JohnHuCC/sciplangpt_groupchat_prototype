from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
from agents.agent_pool_manager import AgentPoolManager, AgentConfig

router = APIRouter()
agent_pool = AgentPoolManager()

# 確保上傳目錄存在
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pydantic models for request/response validation
class AgentCreate(BaseModel):
    name: str
    description: str
    base_prompt: str
    query_templates: List[str] = []

class AgentQuery(BaseModel):
    query: str

class AgentResponse(BaseModel):
    name: str
    description: str
    created_at: str
    type: Optional[str] = None

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections.values():
            await connection.send_json(message)

ws_manager = WebSocketManager()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            # 處理接收到的消息
            if data.get("type") == "query":
                agent_name = data.get("agent_name")
                query = data.get("query")
                
                agent = await agent_pool.get_agent(agent_name)
                if agent:
                    response = await agent.process_query(query)
                    await websocket.send_json(response)
                else:
                    await websocket.send_json({"error": "Agent not found"})
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        ws_manager.disconnect(client_id)

@router.get("/api/agents")
async def list_agents():
    """獲取所有 agents 列表"""
    try:
        # 這裡添加了 await
        agents = await agent_pool.list_agents()
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/api/agents")
async def create_agent(
    name: str = Form(...),
    description: str = Form(...),
    base_prompt: str = Form(...),
    files: List[UploadFile] = File(None),
    query_templates: List[str] = Form([])
):
    """創建新的 agent"""
    try:
        file_paths = []
        if files:
            for file in files:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                content = await file.read()
                with open(filepath, "wb") as f:
                    f.write(content)
                file_paths.append(filepath)
        
        # 創建配置對象
        config = AgentConfig(
            name=name,
            description=description,
            base_prompt=base_prompt,
            query_templates=query_templates
        )
        
        try:
            # 創建 agent
            result = await agent_pool.create_agent(
                config=config,
                knowledge_files=file_paths
            )
            return result
        finally:
            # 清理臨時文件
            for filepath in file_paths:
                if os.path.exists(filepath):
                    os.remove(filepath)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/agents/{name}")
async def get_agent(name: str):
    """獲取特定 agent 的詳細信息"""
    try:
        agent = await agent_pool.get_agent(name)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/agents/{name}/query")
async def query_agent(name: str, query: AgentQuery):
    """使用特定 agent 進行查詢"""
    try:
        agent = await agent_pool.get_agent(name)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        response = await agent.process_query(query.query)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uploads/{filename}")
async def download_file(filename: str):
    """提供上傳文件的下載"""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)