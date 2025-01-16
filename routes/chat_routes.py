from fastapi import (
    APIRouter, 
    WebSocket, 
    WebSocketDisconnect, 
    status,
    Request,
    HTTPException
)
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
from datetime import datetime
from agents.agent_pool_manager import AgentPoolManager
from agents.agent_manager import AgentManager
from pydantic import BaseModel
import logging
import json
import asyncio
from starlette.websockets import WebSocketState
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()
agent_pool = AgentPoolManager()
agent_manager = AgentManager()
templates = Jinja2Templates(directory="templates")

# Pydantic Models
class ChatRoomCreate(BaseModel):
    name: str
    agent_names: List[str]

class ChatMessage(BaseModel):
    message: str

class ChatRoom(BaseModel):
    id: str
    name: str
    agent_names: List[str]
    created_at: str

class Message(BaseModel):
    id: str
    sender: str
    content: str
    timestamp: str
    error: Optional[str] = None
    used_knowledge: Optional[List[Dict[str, Any]]] = None

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        try:
            await websocket.accept()
            if room_id not in self.active_connections:
                self.active_connections[room_id] = []
            self.active_connections[room_id].append(websocket)
            logger.info(f"WebSocket connected to room {room_id}")
        except Exception as e:
            logger.error(f"Error accepting connection: {str(e)}")
            raise

    def disconnect(self, websocket: WebSocket, room_id: str):
        try:
            if room_id in self.active_connections:
                if websocket in self.active_connections[room_id]:
                    self.active_connections[room_id].remove(websocket)
                    logger.info(f"WebSocket disconnected from room {room_id}")
                if not self.active_connections[room_id]:
                    del self.active_connections[room_id]
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")

    async def broadcast(self, message: Dict, room_id: str):
        if room_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[room_id]:
                try:
                    if connection.client_state == WebSocketState.CONNECTED:
                        await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting message: {str(e)}")
                    dead_connections.append(connection)
            
            # Remove dead connections
            for dead in dead_connections:
                self.disconnect(dead, room_id)

manager = ConnectionManager()

# 內存存儲
chat_rooms = {}
chat_history = {}

async def handle_chat_message(websocket: WebSocket, room_id: str, data: Dict):
    """處理聊天消息"""
    try:
        # 創建用戶消息
        user_message = {
            "id": str(uuid.uuid4()),
            "sender": "user",
            "content": data.get("message", ""),
            "timestamp": datetime.now().isoformat()
        }
        chat_history[room_id].append(user_message)

        await manager.broadcast({
            "type": "message",
            "message": user_message
        }, room_id)

        # 獲取房間信息
        room = chat_rooms.get(room_id)
        if not room:
            logger.error(f"Room {room_id} not found")
            return

        # 獲取並註冊所有 agents
        available_agents = []
        for agent_data in room["agents"]:
            agent = await agent_pool.get_agent_instance(agent_data["name"])
            if agent:
                agent_manager.register_agent(agent)
                available_agents.append(agent)

        if not available_agents:
            logger.error("No valid agents found")
            return

        # 發送處理開始消息
        processing_start = {
            "id": str(uuid.uuid4()),
            "sender": "system",
            "content": "開始分析訊息並決定處理順序...",
            "timestamp": datetime.now().isoformat(),
            "status": "processing"
        }
        await manager.broadcast({
            "type": "message",
            "message": processing_start
        }, room_id)

        # 使用 AgentManager 處理消息
        context = {
            "room_id": room_id,
            "history": chat_history[room_id]
        }

        # 使用第一個 agent 作為起始點
        initial_agent = available_agents[0].name
        result = await agent_manager.process_message(
            data.get("message", ""),
            initial_agent,
            context
        )

        # 顯示決定的處理順序
        if "processing_sequence" in result:
            sequence_message = {
                "id": str(uuid.uuid4()),
                "sender": "system",
                "content": f"處理順序: {' -> '.join(result['processing_sequence'])}",
                "timestamp": datetime.now().isoformat(),
                "status": "processing"
            }
            await manager.broadcast({
                "type": "message",
                "message": sequence_message
            }, room_id)

        # 顯示處理過程
        if "result" in result and "message_trail" in result["result"]:
            for step in result["result"]["message_trail"]:
                trail_message = {
                    "id": str(uuid.uuid4()),
                    "sender": step["agent_name"],
                    "content": f"{step['output']}",
                    "timestamp": step["timestamp"],
                    "is_trail": True
                }
                await manager.broadcast({
                    "type": "message",
                    "message": trail_message
                }, room_id)

                # 如果有下一個 agent，顯示轉發訊息
                if step.get("next_agent"):
                    transfer_message = {
                        "id": str(uuid.uuid4()),
                        "sender": "system",
                        "content": f"↓ 轉發給 {step['next_agent']} ↓",
                        "timestamp": step["timestamp"],
                        "is_trail": True
                    }
                    await manager.broadcast({
                        "type": "message",
                        "message": transfer_message
                    }, room_id)

        # 發送最終結果
        if "result" in result and "content" in result["result"]:
            final_message = {
                "id": str(uuid.uuid4()),
                "sender": "system",
                "content": result["result"]["content"],
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            chat_history[room_id].append(final_message)
            await manager.broadcast({
                "type": "message",
                "message": final_message
            }, room_id)

        # 發送處理完成消息
        processing_complete = {
            "id": str(uuid.uuid4()),
            "sender": "system",
            "content": "處理完成",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        await manager.broadcast({
            "type": "message",
            "message": processing_complete
        }, room_id)

    except Exception as e:
        logger.error(f"Error handling chat message: {str(e)}")
        error_message = {
            "id": str(uuid.uuid4()),
            "sender": "system",
            "content": f"處理訊息時發生錯誤: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }
        await manager.broadcast({
            "type": "message",
            "message": error_message
        }, room_id)

# WebSocket endpoint
@router.websocket("/chat/rooms/{room_id}/ws")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    """WebSocket 端點"""
    logger.info(f"WebSocket connection attempt for room {room_id}")
    try:
        if room_id not in chat_rooms:
            logger.error(f"Room {room_id} not found")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await manager.connect(websocket, room_id)

        try:
            while True:
                data = await websocket.receive_json()
                message_type = data.get("type")
                logger.info(f"Received message type: {message_type}")

                if message_type == "chat_message":
                    await handle_chat_message(websocket, room_id, data)
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown message type"
                    })

        except WebSocketDisconnect:
            manager.disconnect(websocket, room_id)
            logger.info(f"WebSocket disconnected from room {room_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {str(e)}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            manager.disconnect(websocket, room_id)

    except Exception as e:
        logger.error(f"Error setting up WebSocket: {str(e)}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

# REST endpoints
@router.get("/chat")
async def chat_room(request: Request):
    """Render chat room page"""
    return templates.TemplateResponse("chat.html", {"request": request})

@router.post("/api/chat/rooms")
async def create_chat_room(room_data: ChatRoomCreate):
    """Create a new chat room"""
    try:
        logger.info(f"Creating chat room with data: {room_data.dict()}")
        
        if not room_data.agent_names:
            logger.error("No agents specified for chat room")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one agent is required"
            )

        valid_agents = []
        for agent_name in room_data.agent_names:
            try:
                agent = await agent_pool.get_agent(agent_name)
                if agent is None:
                    logger.error(f"Agent not found: {agent_name}")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Agent {agent_name} not found"
                    )
                valid_agents.append({
                    "name": agent_name,
                    "config": agent
                })
            except Exception as e:
                logger.error(f"Error getting agent {agent_name}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error getting agent {agent_name}"
                )

        room_id = str(len(chat_rooms) + 1)
        room = {
            "id": room_id,
            "name": room_data.name,
            "agent_names": room_data.agent_names,
            "created_at": datetime.now().isoformat(),
            "agents": valid_agents
        }

        chat_rooms[room_id] = room
        chat_history[room_id] = []

        logger.info(f"Successfully created chat room {room_id}")
        return room

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating chat room: {str(e)}")
        if 'room_id' in locals() and room_id in chat_rooms:
            chat_rooms.pop(room_id, None)
            chat_history.pop(room_id, None)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat room: {str(e)}"
        )

@router.get("/api/chat/rooms")
async def list_chat_rooms():
    """Get list of all chat rooms"""
    return list(chat_rooms.values())

@router.get("/api/chat/rooms/{room_id}")
async def get_chat_room(room_id: str):
    """Get specific chat room info"""
    if room_id not in chat_rooms:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat room not found"
        )
    return chat_rooms[room_id]

@router.get("/api/chat/rooms/{room_id}/messages")
async def get_chat_history(room_id: str):
    """Get chat room message history"""
    if room_id not in chat_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat room not found"
        )
    return chat_history[room_id]

@router.delete("/api/chat/rooms/{room_id}")
async def delete_chat_room(room_id: str):
    """Delete a chat room"""
    if room_id not in chat_rooms:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat room not found"
        )

    if room_id in manager.active_connections:
        for connection in manager.active_connections[room_id]:
            await connection.close(code=status.WS_1000_NORMAL_CLOSURE)
    
    del chat_rooms[room_id]
    del chat_history[room_id]

    return {"message": f"Chat room {room_id} deleted successfully"}