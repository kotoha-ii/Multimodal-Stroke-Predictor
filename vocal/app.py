from flask import Flask, render_template, request, jsonify
from core.audio_processor import analyze_audio
import os

# app = Flask(__name__, template_folder='web/templates')

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
print(f"Template directory: {template_dir}")
print(f"Static directory: {static_dir}")  # 添加静态文件夹路径的调试输出
print(f"Template directory: {template_dir}")  # 调试输出
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(filepath)
    
    try:
        results = analyze_audio(filepath)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)