from flask import Blueprint, jsonify, request, render_template
from typing import List, Dict, Any
import asyncio
from datetime import datetime
from agents.agent_pool_manager import AgentPoolManager

chat_bp = Blueprint('chat', __name__)
agent_pool = AgentPoolManager()

# 簡單的內存存儲聊天記錄
# 在實際應用中，應該使用資料庫
chat_history = {}
chat_rooms = {}

async def process_agent_query(agent_name: str, query: str) -> Dict[str, Any]:
    """非同步處理單個 agent 的查詢"""
    try:
        agent = agent_pool.get_agent(agent_name)
        if not agent:
            return {
                'error': f'Agent {agent_name} not found',
                'agent_name': agent_name
            }
            
        response = await agent.process_query_async(query)
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

@chat_bp.route('/chat')
def chat_room():
    """渲染聊天室頁面"""
    return render_template('chat.html')

@chat_bp.route('/api/chat/rooms', methods=['POST'])
def create_chat_room():
    """創建新的聊天室"""
    try:
        data = request.json
        room_name = data.get('name')
        agent_names = data.get('agent_names', [])
        
        if not room_name:
            return jsonify({'error': 'Room name is required'}), 400
            
        if not agent_names:
            return jsonify({'error': 'At least one agent is required'}), 400
            
        # 驗證所有 agent 是否存在
        for agent_name in agent_names:
            if not agent_pool.get_agent(agent_name):
                return jsonify({'error': f'Agent {agent_name} not found'}), 404
        
        room_id = str(len(chat_rooms) + 1)  # 簡單的 ID 生成
        chat_rooms[room_id] = {
            'id': room_id,
            'name': room_name,
            'agent_names': agent_names,
            'created_at': datetime.now().isoformat()
        }
        chat_history[room_id] = []
        
        return jsonify(chat_rooms[room_id])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/api/chat/rooms', methods=['GET'])
def list_chat_rooms():
    """獲取所有聊天室列表"""
    return jsonify(list(chat_rooms.values()))

@chat_bp.route('/api/chat/rooms/<room_id>', methods=['GET'])
def get_chat_room(room_id):
    """獲取特定聊天室信息"""
    if room_id not in chat_rooms:
        return jsonify({'error': 'Chat room not found'}), 404
    return jsonify(chat_rooms[room_id])

@chat_bp.route('/api/chat/rooms/<room_id>/messages', methods=['GET'])
def get_chat_history(room_id):
    """獲取聊天室的消息歷史"""
    if room_id not in chat_history:
        return jsonify({'error': 'Chat room not found'}), 404
    return jsonify(chat_history[room_id])

@chat_bp.route('/api/chat/rooms/<room_id>/messages', methods=['POST'])
async def send_message(room_id):
    """發送消息到聊天室"""
    try:
        if room_id not in chat_rooms:
            return jsonify({'error': 'Chat room not found'}), 404
            
        data = request.json
        message = data.get('message')
        if not message:
            return jsonify({'error': 'Message is required'}), 400
            
        # 記錄用戶消息
        user_message = {
            'id': str(len(chat_history[room_id]) + 1),
            'sender': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        }
        chat_history[room_id].append(user_message)
        
        # 獲取聊天室的 agents
        agent_names = chat_rooms[room_id]['agent_names']
        
        # 非同步處理所有 agent 的回應
        tasks = [
            process_agent_query(agent_name, message)
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
        
        return jsonify({
            'user_message': user_message,
            'agent_responses': agent_responses
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/api/chat/rooms/<room_id>', methods=['DELETE'])
def delete_chat_room(room_id):
    """刪除聊天室"""
    if room_id not in chat_rooms:
        return jsonify({'error': 'Chat room not found'}), 404
        
    del chat_rooms[room_id]
    del chat_history[room_id]
    
    return jsonify({'message': f'Chat room {room_id} deleted successfully'})