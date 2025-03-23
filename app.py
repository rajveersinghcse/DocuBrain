from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from rag_system import RagSystem
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_flask_app")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

rag = RagSystem()

ALLOWED_EXTENSIONS = {
    'text': ['.txt', '.md', '.json'],
    'image': ['.jpg', '.jpeg', '.png', '.gif'],
    'video': ['.mp4', '.avi', '.mov', '.mkv'],
    'audio': ['.mp3', '.wav', '.ogg', '.flac'],
    'pdf': ['.pdf'],
    'tabular': ['.csv', '.xlsx', '.xls']
}

def allowed_file(filename):
    return '.' in filename and any(
        filename.lower().endswith(ext) for exts in ALLOWED_EXTENSIONS.values() for ext in exts
    )

def get_file_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    for file_type, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return file_type
    return "unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    query = request.form.get('query', '')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            result = rag.process_file(file_path, query)
            result['file_type_human'] = get_file_type(filename)
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/supported-types')
def supported_types():
    return jsonify({
        "types": ALLOWED_EXTENSIONS
    })

@app.route('/clean')
def cleanup():
    try:
        rag.cleanup()
        return jsonify({"message": "Cleanup successful"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)