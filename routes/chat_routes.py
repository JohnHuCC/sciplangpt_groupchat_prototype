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
from pydantic import BaseModel
import logging
import json
import asyncio
from starlette.websockets import WebSocketState
import uuid 

# 設置日誌
logger = logging.getLogger(__name__)

router = APIRouter()
agent_pool = AgentPoolManager()
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

async def process_agent_message(
    agent: Any,
    agent_name: str,
    message: str,
    history: List[Dict]
) -> Dict[str, Any]:
    """處理單個 agent 的消息"""
    try:
        logger.info(f"Processing message for agent {agent_name}")
        logger.info(f"Agent type: {type(agent)}")  # 記錄 agent 類型
        logger.info(f"Agent attributes: {vars(agent)}")  # 記錄 agent 屬性
        
        try:
            logger.info(f"Calling agent.process_query with message: {message}")
            response = await agent.process_query(message)
            logger.info(f"Raw response type: {type(response)}")  # 記錄回應類型
            logger.info(f"Raw response content: {response}")

            if not response:
                logger.error(f"Agent {agent_name} returned empty response")
                raise ValueError(f"Empty response from agent {agent_name}")

            formatted_response = {
                "agent_name": agent_name,
                "content": "",
                "used_knowledge": [],
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }

            # 詳細的回應處理
            if isinstance(response, str):
                logger.info(f"Processing string response: {response}")
                formatted_response["content"] = response
            elif isinstance(response, dict):
                logger.info(f"Processing dict response: {response}")
                formatted_response["content"] = response.get("content", "")
                if "metadata" in response and "sources" in response["metadata"]:
                    formatted_response["used_knowledge"] = response["metadata"]["sources"]
                    logger.info(f"Found knowledge sources: {formatted_response['used_knowledge']}")

            # 驗證回應內容
            if not formatted_response["content"]:
                logger.warning(f"Empty content from agent {agent_name}, checking agent configuration")
                # 檢查 agent 配置
                agent_config = await agent.get_configuration()
                logger.info(f"Agent configuration: {agent_config}")
                formatted_response["content"] = f"Agent {agent_name} provided no response content. Please check agent configuration."
                formatted_response["status"] = "error"

            logger.info(f"Final formatted response: {formatted_response}")
            return formatted_response

        except Exception as e:
            logger.exception(f"Error processing message for agent {agent_name}")  # 使用 exception 記錄完整堆疊
            return {
                "agent_name": agent_name,
                "error": True,
                "content": f"Error processing message: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "used_knowledge": []
            }

    except Exception as e:
        logger.exception(f"Critical error in agent {agent_name}")
        return {
            "agent_name": agent_name,
            "error": True,
            "content": f"Critical error in agent {agent_name}: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "used_knowledge": []
        }

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

        # 準備 agent 序列並顯示處理順序
        agent_sequence = []
        agent_names = []
        for agent_data in room["agents"]:
            agent_name = agent_data["name"]
            agent_names.append(agent_name)
            agent = await agent_pool.get_agent_instance(agent_name)
            if agent:
                if agent_sequence:
                    agent_sequence[-1].set_next_agent(agent)
                agent_sequence.append(agent)

        if not agent_sequence:
            logger.error("No valid agents found")
            return

        # 發送處理開始消息
        processing_start = {
            "id": str(uuid.uuid4()),
            "sender": "system",
            "content": f"開始處理訊息...\n處理順序: {' -> '.join(agent_names)}",
            "timestamp": datetime.now().isoformat(),
            "status": "processing"
        }
        await manager.broadcast({
            "type": "message",
            "message": processing_start
        }, room_id)

        # 開始處理
        context = {
            "room_id": room_id,
            "history": chat_history[room_id]
        }
        
        response = await agent_sequence[0].process_query(
            data.get("message", ""),
            context
        )

        # 顯示訊息傳遞過程
        if "message_trail" in response:
            for i, step in enumerate(response["message_trail"]):
                # 顯示當前 agent 收到的輸入
                input_message = {
                    "id": str(uuid.uuid4()),
                    "sender": "system",
                    "content": f"[{step['agent_name']} 接收到的輸入]\n{step['input']}",
                    "timestamp": step["timestamp"],
                    "is_trail": True,
                    "type": "input"
                }
                await manager.broadcast({
                    "type": "message",
                    "message": input_message
                }, room_id)

                # 顯示當前 agent 的處理結果
                output_message = {
                    "id": str(uuid.uuid4()),
                    "sender": "system",
                    "content": f"[{step['agent_name']} 的處理結果]\n{step['output']}",
                    "timestamp": step["timestamp"],
                    "is_trail": True,
                    "type": "output"
                }
                await manager.broadcast({
                    "type": "message",
                    "message": output_message
                }, room_id)

                # 如果不是最後一個 agent，顯示訊息傳遞
                if i < len(response["message_trail"]) - 1:
                    next_agent = response["message_trail"][i + 1]["agent_name"]
                    transfer_message = {
                        "id": str(uuid.uuid4()),
                        "sender": "system",
                        "content": f"↓ 訊息傳遞給 {next_agent} ↓",
                        "timestamp": step["timestamp"],
                        "is_trail": True,
                        "type": "transfer"
                    }
                    await manager.broadcast({
                        "type": "message",
                        "message": transfer_message
                    }, room_id)

        # 發送最終結果
        final_message = {
            "id": str(uuid.uuid4()),
            "sender": agent_sequence[-1].name,
            "content": response.get("content", "沒有回應內容"),
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

    except HTTPException as he:
        logger.error(f"HTTP exception during chat room creation: {str(he)}")
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