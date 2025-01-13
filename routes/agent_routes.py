from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import logging
from agents.agent_pool_manager import AgentPoolManager, AgentConfig
from utils.template_manager import TemplateManager

# 設置日誌
logger = logging.getLogger(__name__)

router = APIRouter()
agent_pool = AgentPoolManager()
template_manager = TemplateManager()

# 確保上傳目錄存在
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pydantic models for request/response validation
class AgentCreate(BaseModel):
    name: str
    description: str
    template_name: str
    parameters: Dict[str, Any] = {}

class AgentQuery(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    name: str
    description: str
    created_at: str
    type: Optional[str] = None
    template_name: Optional[str] = None

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def broadcast(self, message: Dict[str, Any]):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)
        
        # 清理斷開的連接
        for client_id in disconnected_clients:
            self.disconnect(client_id)

ws_manager = WebSocketManager()

# Template 相關路由
@router.get("/api/templates")
async def list_templates():
    """列出所有可用的 agent 模板"""
    try:
        templates = await template_manager.list_templates()
        return JSONResponse(content=templates)
    except Exception as e:
        logger.error(f"Error listing templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/templates/{template_name}")
async def get_template(template_name: str):
    """獲取指定模板的詳細信息"""
    try:
        template = await template_manager.load_template(template_name)
        return JSONResponse(content=template)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading template {template_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            agent_name = data.get("agent_name")
            query = data.get("query")
            parameters = data.get("parameters")

            if not agent_name or not query:
                await websocket.send_json({
                    "error": "Missing required fields (agent_name or query)"
                })
                continue

            try:
                agent = await agent_pool.get_agent(agent_name)
                if not agent:
                    await websocket.send_json({"error": f"Agent {agent_name} not found"})
                    continue

                # 處理查詢
                if parameters:
                    original_parameters = agent.parameters.copy()
                    agent.parameters.update(parameters)

                response = await agent.process_query(query)

                if parameters:
                    agent.parameters = original_parameters

                await websocket.send_json(response)

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                await websocket.send_json({"error": str(e)})

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
    finally:
        ws_manager.disconnect(client_id)

# Agent 管理路由
@router.get("/api/agents")
async def list_agents():
    """獲取所有 agents 列表"""
    try:
        agents = await agent_pool.list_agents()
        return JSONResponse(content=agents)
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/agents")
async def create_agent(
    name: str = Form(...),
    description: str = Form(...),
    template_name: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    """創建新的 agent"""
    try:
        logger.info(f"Creating agent: {name} with template: {template_name}")
        
        # 載入模板
        try:
            template = await template_manager.load_template(template_name)
            if not template:
                raise ValueError(f"Template {template_name} not found")
        except Exception as e:
            logger.error(f"Error loading template: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Template {template_name} not found")

        # 處理上傳的文件
        uploaded_files = []
        if files:
            upload_dir = os.path.join(UPLOAD_FOLDER, name)
            os.makedirs(upload_dir, exist_ok=True)
            
            for file in files:
                try:
                    file_path = os.path.join(upload_dir, file.filename)
                    content = await file.read()
                    with open(file_path, "wb") as f:
                        f.write(content)
                    uploaded_files.append(file_path)
                    logger.info(f"Saved file: {file_path}")
                except Exception as e:
                    logger.error(f"Error saving file {file.filename}: {str(e)}")
                    # 清理已上傳的文件
                    for path in uploaded_files:
                        try:
                            os.remove(path)
                        except:
                            pass
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Error saving file {file.filename}: {str(e)}"
                    )

        # 創建 agent 配置
        try:
            agent_config = AgentConfig(
                name=name,
                description=description,
                template_name=template_name,
                type=template.type,  # 從模板中獲取類型
                parameters={}  # 可以根據需要添加參數
            )
            
            # 創建 agent
            agent = await agent_pool.create_agent(agent_config, uploaded_files)
            if not agent:
                raise ValueError("Failed to create agent")
                
            return JSONResponse(content=agent)

        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            # 清理上傳的文件
            if uploaded_files:
                upload_dir = os.path.join(UPLOAD_FOLDER, name)
                if os.path.exists(upload_dir):
                    import shutil
                    shutil.rmtree(upload_dir)
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/agents/{name}")
async def get_agent(name: str):
    """獲取特定 agent 的詳細信息"""
    try:
        agent = await agent_pool.get_agent(name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {name} not found")
        return JSONResponse(content=agent.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/agents/{name}")
async def delete_agent(name: str):
    """刪除指定的 agent"""
    try:
        success = await agent_pool.delete_agent(name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Agent {name} not found")

        # 清理相關文件
        upload_dir = os.path.join(UPLOAD_FOLDER, name)
        if os.path.exists(upload_dir):
            import shutil
            shutil.rmtree(upload_dir)

        return JSONResponse(content={"message": f"Agent {name} deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/agents/{name}")
async def update_agent(name: str, updates: AgentCreate):
    """更新 agent 配置"""
    try:
        agent = await agent_pool.update_agent(name, updates.dict())
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {name} not found")
        return JSONResponse(content=agent.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/uploads/{filename}")
async def download_file(filename: str):
    """下載上傳的文件"""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    return FileResponse(file_path)