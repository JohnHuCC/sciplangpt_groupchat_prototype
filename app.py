from flask import Flask, request, jsonify, render_template, Response
import os
import sys
import glob
import queue
import create_embedding
import humanize
from agents import QuestionBasedResearchAgent

app = Flask(__name__)

# 配置 LLM
llm_config = {
    "model": "gpt-4",
    "temperature": 0.7
}

status_queue = queue.Queue()

def update_status(status):
    status_queue.put(status)

def get_knowledge_base_stats():
    """獲取知識庫統計信息"""
    try:
        texts, sources, metadata = create_embedding.load_index()
        files = glob.glob(os.path.join(create_embedding.DOCS_DIR, '*'))
        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        
        return {
            'file_count': len(set(sources)),
            'total_chunks': len(texts),
            'total_size': humanize.naturalsize(total_size),
            'files': [os.path.basename(f) for f in files]
        }
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return {
            'file_count': 0,
            'total_chunks': 0,
            'total_size': '0 B',
            'files': []
        }

@app.route('/')
def home():
    stats = get_knowledge_base_stats()
    return render_template('index.html', stats=stats)

@app.route('/view_knowledge_base')
def view_knowledge_base():
    texts, sources, metadata = create_embedding.load_index()
    
    knowledge_base = {}
    for text, source in zip(texts, sources):
        filename = os.path.basename(source)
        if filename not in knowledge_base:
            knowledge_base[filename] = []
        knowledge_base[filename].append(text[:200] + '...' if len(text) > 200 else text)
    
    return render_template('knowledge_base.html', knowledge_base=knowledge_base, stats=get_knowledge_base_stats())

@app.route('/api/knowledge_base_stats')
def get_stats():
    return jsonify(get_knowledge_base_stats())

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    try:
        data = request.json
        research_area = data.get('research_area')
        
        if not research_area:
            return jsonify({'error': 'Research area is required'})

        # 創建 QuestionBasedResearchAgent 實例
        agent = QuestionBasedResearchAgent(llm_config)
        
        # 使用 agent 生成研究計畫
        result = agent.generate_research_plan(research_area)
        
        # 準備查詢使用的知識內容
        used_knowledge = []
        for question, query_result in result['knowledge_results'].items():
            if "texts" in query_result and "sources" in query_result:
                for text, source in zip(query_result["texts"], query_result["sources"]):
                    used_knowledge.append({
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'source': os.path.basename(source)
                    })
        
        # 格式化回應
        return jsonify({
            'research_area': research_area,
            'research_question': result['questions'],
            'research_prompt': "Based on the generated questions and knowledge base query results",
            'research_plan': result['research_plan'],
            'used_knowledge': used_knowledge
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload_to_knowledge', methods=['POST'])
def upload_to_knowledge():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not file.filename.lower().endswith(('.pdf', '.txt', '.docx')):
        return jsonify({'error': 'File type not allowed'})
    
    try:
        os.makedirs(create_embedding.DOCS_DIR, exist_ok=True)
        filepath = os.path.join(create_embedding.DOCS_DIR, file.filename)
        file.save(filepath)
        create_embedding.process_files(create_embedding.DOCS_DIR)
        
        return jsonify({'message': 'File successfully added to knowledge base'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)