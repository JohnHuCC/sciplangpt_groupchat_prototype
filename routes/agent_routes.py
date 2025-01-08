from flask import Blueprint, request, jsonify, send_from_directory
import os
from agents.agent_pool_manager import AgentPoolManager

agent_bp = Blueprint('agent', __name__)
agent_pool = AgentPoolManager()

# 確保上傳目錄存在
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@agent_bp.route('/api/agents', methods=['GET'])
def list_agents():
    """獲取所有 agents 列表"""
    try:
        agents = agent_pool.list_agents()
        return jsonify(agents)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@agent_bp.route('/api/agents', methods=['POST'])
def create_agent():
    """創建新的 agent"""
    try:
        # 處理文件上傳
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
            
        files = request.files.getlist('files[]')
        file_paths = []
        
        for file in files:
            if file.filename:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                file_paths.append(filepath)
        
        # 獲取其他數據
        data = request.form
        required_fields = ['name', 'description', 'base_prompt']
        
        # 驗證必需字段
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # 處理 query_templates
        query_templates = []
        i = 0
        while f'query_template_{i}' in data:
            template = data[f'query_template_{i}']
            if template.strip():  # 只添加非空模板
                query_templates.append(template)
            i += 1
        
        # 創建 agent
        config = agent_pool.create_agent(
            name=data['name'],
            description=data['description'],
            knowledge_files=file_paths,
            base_prompt=data['base_prompt'],
            query_templates=query_templates
        )
        
        # 清理臨時文件
        for filepath in file_paths:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify(config)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@agent_bp.route('/api/agents/<name>', methods=['GET'])
def get_agent(name):
    """獲取特定 agent 的詳細信息"""
    agent = agent_pool.get_agent(name)
    if agent is None:
        return jsonify({'error': 'Agent not found'}), 404
    return jsonify(agent.to_dict())

@agent_bp.route('/api/agents/<name>', methods=['DELETE'])
def delete_agent(name):
    """刪除特定 agent"""
    success = agent_pool.delete_agent(name)
    if not success:
        return jsonify({'error': 'Agent not found'}), 404
    return jsonify({'message': f'Agent {name} deleted successfully'})

@agent_bp.route('/api/agents/<name>/query', methods=['POST'])
def query_agent(name):
    """使用特定 agent 進行查詢"""
    try:
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({'error': 'Query is required'}), 400
            
        agent = agent_pool.get_agent(name)
        if agent is None:
            return jsonify({'error': 'Agent not found'}), 404
            
        response = agent.process_query(query)
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@agent_bp.route('/uploads/<path:filename>')
def download_file(filename):
    """提供上傳文件的下載"""
    return send_from_directory(UPLOAD_FOLDER, filename)