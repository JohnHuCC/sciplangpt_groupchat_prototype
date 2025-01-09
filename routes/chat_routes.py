from fastapi import APIRouter, Request, HTTPException, WebSocket
from fastapi.responses import JSONResponse  # 如果需要手動回傳 JSON 的話
from typing import List, Dict, Any, Optional
from datetime import datetime
from agents.agent_pool_manager import AgentPoolManager
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio  # 添加 asyncio 導入，因為我們使用了 gather

router = APIRouter()
agent_pool = AgentPoolManager()
templates = Jinja2Templates(directory="templates")

# 其餘代碼保持不變...

# 簡單的內存存儲聊天記錄
chat_history = {}
chat_rooms = {}

class ChatRoomCreate(BaseModel):
    name: str
    agent_names: List[str]

class ChatMessage(BaseModel):
    message: str

# 添加更多 Pydantic 模型來規範資料結構
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

class ChatResponse(BaseModel):
    user_message: Message
    agent_responses: List[Dict[str, Any]] 

async def process_agent_query(agent_name: str, query: str) -> Dict[str, Any]:
    """非同步處理單個 agent 的查詢"""
    try:
        # 使用 await 調用異步方法
        agent = await agent_pool.get_agent(agent_name)
        if not agent:
            return {
                'error': f'Agent {agent_name} not found',
                'agent_name': agent_name
            }
            
        response = await agent.process_query(query)
        return {
            'agent_name': agent_name,
            'response': response['response'],
            'used_knowledge': response.get('used_knowledge', []),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'error': str(e),
            'agent_name': agent_name,
            'timestamp': datetime.now().isoformat()
        }

@router.get("/chat")
async def chat_room(request: Request):
    """渲染聊天室頁面"""
    return templates.TemplateResponse("chat.html", {"request": request})

@router.post("/api/chat/rooms")
async def create_chat_room(room_data: ChatRoomCreate):
    """創建新的聊天室"""
    try:
        if not room_data.agent_names:
            raise HTTPException(status_code=400, detail="At least one agent is required")
            
        # 驗證所有 agent 是否存在
        for agent_name in room_data.agent_names:
            # 添加 await
            agent = await agent_pool.get_agent(agent_name)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        room_id = str(len(chat_rooms) + 1)
        chat_rooms[room_id] = {
            'id': room_id,
            'name': room_data.name,
            'agent_names': room_data.agent_names,
            'created_at': datetime.now().isoformat()
        }
        chat_history[room_id] = []
        
        return chat_rooms[room_id]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/chat/rooms")
async def list_chat_rooms():
    """獲取所有聊天室列表"""
    return list(chat_rooms.values())

@router.get("/api/chat/rooms/{room_id}")
async def get_chat_room(room_id: str):
    """獲取特定聊天室信息"""
    if room_id not in chat_rooms:
        raise HTTPException(status_code=404, detail="Chat room not found")
    return chat_rooms[room_id]

@router.get("/api/chat/rooms/{room_id}/messages")
async def get_chat_history(room_id: str):
    """獲取聊天室的消息歷史"""
    if room_id not in chat_history:
        raise HTTPException(status_code=404, detail="Chat room not found")
    return chat_history[room_id]

@router.post("/api/chat/rooms/{room_id}/messages")
async def send_message(room_id: str, message: ChatMessage):
    """發送消息到聊天室"""
    try:
        if room_id not in chat_rooms:
            raise HTTPException(status_code=404, detail="Chat room not found")
            
        # 記錄用戶消息
        user_message = {
            'id': str(len(chat_history[room_id]) + 1),
            'sender': 'user',
            'content': message.message,
            'timestamp': datetime.now().isoformat()
        }
        chat_history[room_id].append(user_message)
        
        # 獲取聊天室的 agents
        agent_names = chat_rooms[room_id]['agent_names']
        
        # 非同步處理所有 agent 的回應
        tasks = [
            process_agent_query(agent_name, message.message)
            for agent_name in agent_names
        ]
        
        agent_responses = await asyncio.gather(*tasks)
        
        # 記錄 agent 回應
        for response in agent_responses:
            agent_message = {
                'id': str(len(chat_history[room_id]) + 1),
                'sender': response['agent_name'],
                'content': response.get('response', ''),
                'error': response.get('error'),
                'used_knowledge': response.get('used_knowledge', []),
                'timestamp': response['timestamp']
            }
            chat_history[room_id].append(agent_message)
        
        return {
            'user_message': user_message,
            'agent_responses': agent_responses
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/chat/rooms/{room_id}")
async def delete_chat_room(room_id: str):
    """刪除聊天室"""
    if room_id not in chat_rooms:
        raise HTTPException(status_code=404, detail="Chat room not found")
        
    del chat_rooms[room_id]
    del chat_history[room_id]
    
    return {"message": f"Chat room {room_id} deleted successfully"}