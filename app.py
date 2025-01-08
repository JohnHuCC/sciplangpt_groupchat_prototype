from flask import Flask, render_template
import os
from routes.agent_routes import agent_bp
from routes.chat_routes import chat_bp

app = Flask(__name__)

# 確保存在上傳文件目錄
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 註冊藍圖
app.register_blueprint(agent_bp)
app.register_blueprint(chat_bp)

@app.route('/')
def home():
    """首頁"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)