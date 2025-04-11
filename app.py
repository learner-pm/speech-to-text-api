from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from src.speech_to_text import SpeechToText
from src.deepseek_api import DeepSeekAPI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 初始化语音转文本服务
speech_to_text = SpeechToText()

# 初始化 DeepSeek API 客户端
deepseek_api = DeepSeekAPI()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """处理音频文件上传并返回识别结果"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        result = speech_to_text.transcribe(file)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """处理与 DeepSeek API 的对话请求"""
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': 'No messages provided'}), 400
            
        messages = data['messages']
        model = data.get('model', 'deepseek-chat')
        temperature = data.get('temperature', 0.7)
        
        response = deepseek_api.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature
        )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/embedding', methods=['POST'])
def embedding():
    """处理文本嵌入请求"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        model = data.get('model', 'deepseek-embedding')
        
        response = deepseek_api.text_embedding(
            text=text,
            model=model
        )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in embedding endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 