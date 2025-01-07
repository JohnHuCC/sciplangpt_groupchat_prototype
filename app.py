from flask import Flask, request, jsonify, render_template, Response
import os
import sys
import glob
import json
from werkzeug.utils import secure_filename
from agents import QuestionBasedResearchAgent
import create_embedding
import uuid
from datetime import datetime

app = Flask(__name__)

# 配置
AGENTS_DIR = "agents"
AGENTS_CONFIG_FILE = os.path.join(AGENTS_DIR, "agents_config.json")

def init_agents_directory():
    """初始化 agents 目錄和配置文件"""
    os.makedirs(AGENTS_DIR, exist_ok=True)
    if not os.path.exists(AGENTS_CONFIG_FILE):
        with open(AGENTS_CONFIG_FILE, 'w') as f:
            json.dump({"agents": []}, f)

def load_agents_config():
    """載入 agents 配置"""
    with open(AGENTS_CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_agents_config(config):
    """保存 agents 配置"""
    with open(AGENTS_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

@app.route('/api/agents')
def get_agents():
    """獲取所有可用的 agents"""
    try:
        config = load_agents_config()
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)})

# 在 app.py 中修改 create_agent 路由
@app.route('/api/create_agent', methods=['POST'])
def create_agent():
    """創建新的 agent"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'})
    
    files = request.files.getlist('files[]')
    if not files or not any(file.filename for file in files):
        return jsonify({'error': 'No selected files'})

    try:
        # 創建新的 agent 目錄
        agent_id = str(uuid.uuid4())
        agent_dir = os.path.join(AGENTS_DIR, agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        
        # 保存上傳的文件
        file_paths = []
        error_files = []
        
        for file in files:
            if file.filename:
                if not file.filename.lower().endswith(('.pdf', '.txt', '.docx')):
                    error_files.append({
                        'name': file.filename,
                        'error': 'File type not allowed'
                    })
                    continue
                    
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(agent_dir, filename)
                    file.save(filepath)
                    file_paths.append(filepath)
                except Exception as e:
                    error_files.append({
                        'name': file.filename,
                        'error': str(e)
                    })
        
        if file_paths:
            # 為這個 agent 創建 embedding
            success = create_embedding.process_files_for_agent(agent_dir)
            
            if not success:
                return jsonify({'error': 'Failed to create embeddings for the agent'})
            
            # 更新 agents 配置
            config = load_agents_config()
            config['agents'].append({
                'id': agent_id,
                'name': f"Agent_{len(config['agents']) + 1}",
                'files': [os.path.basename(f) for f in file_paths],
                'created_at': datetime.now().isoformat()
            })
            save_agents_config(config)
            
            response = {
                'message': 'Agent created successfully',
                'agent_id': agent_id,
                'uploaded_files': [os.path.basename(f) for f in file_paths]
            }
            
            if error_files:
                response['errors'] = error_files
                
            return jsonify(response)
        else:
            return jsonify({'error': 'No valid files were uploaded'})
        
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/upload_to_knowledge', methods=['POST'])
def upload_to_knowledge():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'})
    
    files = request.files.getlist('files[]')
    if not files or not any(file.filename for file in files):
        return jsonify({'error': 'No selected files'})
    
    uploaded_files = []
    error_files = []
    
    try:
        # 確保目錄存在
        os.makedirs(create_embedding.DOCS_DIR, exist_ok=True)
        
        # 處理每個上傳的檔案
        for file in files:
            if file.filename:
                # 檢查檔案類型
                if not file.filename.lower().endswith(('.pdf', '.txt', '.docx')):
                    error_files.append({
                        'name': file.filename,
                        'error': 'File type not allowed'
                    })
                    continue
                
                try:
                    # 保存檔案
                    filepath = os.path.join(create_embedding.DOCS_DIR, file.filename)
                    file.save(filepath)
                    uploaded_files.append(file.filename)
                except Exception as e:
                    error_files.append({
                        'name': file.filename,
                        'error': str(e)
                    })
        
        if uploaded_files:
            # 處理所有成功上傳的檔案
            create_embedding.process_files(create_embedding.DOCS_DIR)
            
        response = {
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'uploaded_files': uploaded_files
        }
        
        if error_files:
            response['errors'] = error_files
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/api/generate_plan', methods=['POST'])
def generate_plan():
    """使用選定的 agent 生成研究計畫"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        agent_id = data.get('agent_id')
        research_area = data.get('research_area')
        
        print(f"Received request - agent_id: {agent_id}, research_area: {research_area}")
        
        if not agent_id or not research_area:
            return jsonify({'error': 'Agent ID and research area are required'}), 400
            
        # 檢查 agent 是否存在
        config = load_agents_config()
        agent_config = next((a for a in config['agents'] if a['id'] == agent_id), None)
        if not agent_config:
            return jsonify({'error': 'Agent not found'}), 404
            
        # 初始化 agent
        agent = QuestionBasedResearchAgent({
            "model": "gpt-4",
            "temperature": 0.7
        })
        
        # 設置知識庫
        agent_dir = os.path.join(AGENTS_DIR, agent_id)
        agent.set_knowledge_base(agent_dir)
        
        # 生成研究計畫
        result = agent.generate_research_plan(research_area)
        
        return jsonify({
            'research_area': research_area,
            'questions': result['questions'],
            'knowledge_results': result['knowledge_results'],
            'research_plan': result['research_plan']
        })
        
    except Exception as e:
        print(f"Error in generate_plan: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500   

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    init_agents_directory()
    app.run(debug=True)