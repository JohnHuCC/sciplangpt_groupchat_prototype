# agent_routes.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import logging
from agents.agent_pool_manager import AgentPoolManager, AgentConfig, AgentCreateBase
from utils.template_manager import TemplateManager
import json

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()
agent_pool = AgentPoolManager()
template_manager = TemplateManager()

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pydantic models for request/response validation
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
        
        for client_id in disconnected_clients:
            self.disconnect(client_id)

ws_manager = WebSocketManager()

# Template related routes
@router.get("/api/templates")
async def list_templates():
    """List all available agent templates"""
    try:
        templates = await template_manager.list_templates()
        return JSONResponse(content=templates)
    except Exception as e:
        logger.error(f"Error listing templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/templates/{template_name}")
async def get_template(template_name: str):
    """Get detailed information for specified template"""
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
                agent = await agent_pool.get_agent_instance(agent_name)
                if not agent:
                    await websocket.send_json({"error": f"Agent {agent_name} not found"})
                    continue

                # Handle query
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

# Agent CRUD operations
@router.get("/api/agents")
async def list_agents():
    """List all agents"""
    try:
        return await agent_pool.list_agents()
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/agents")
async def create_agent(
    name: str = Form(...),
    description: str = Form(...),
    template_name: str = Form(...),
    parameters: str = Form("{}"),  # JSON string
    files: List[UploadFile] = File(default=[])
):
    """Create new agent"""
    try:
        # Convert parameters from JSON string to dict
        parameters_dict = json.loads(parameters)
        
        # Create AgentCreateBase object
        agent_data = AgentCreateBase(
            name=name,
            description=description,
            template_name=template_name,
            parameters=parameters_dict
        )

        # Save uploaded files
        knowledge_files = []
        for file in files:
            file_path = f"{UPLOAD_FOLDER}/{file.filename}"
            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                knowledge_files.append(file_path)
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {str(e)}")
                continue
        
        # Create agent
        result = await agent_pool.create_agent(agent_data, knowledge_files)
        return result
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating agent: {str(e)}"
        )

@router.get("/api/agents/{name}")
async def get_agent(name: str):
    """Get detailed information for specified agent"""
    try:
        agent = await agent_pool.get_agent(name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {name} not found")
        return JSONResponse(content=agent)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/agents/{name}")
async def delete_agent(name: str):
    """Delete specified agent"""
    try:
        success = await agent_pool.delete_agent(name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Agent {name} not found")

        # Clean up related files
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
async def update_agent(name: str, updates: AgentCreateBase):
    """Update agent configuration"""
    try:
        agent = await agent_pool.update_agent(name, updates.dict())
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {name} not found")
        return JSONResponse(content=agent)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/uploads/{filename}")
async def download_file(filename: str):
    """Download uploaded file"""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    return FileResponse(file_path)