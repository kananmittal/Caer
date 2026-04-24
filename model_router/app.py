from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mimetypes
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS so the separate frontend can communicate with the backend

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def preprocess_and_route(input_data, modality):
    """
    Data Preprocessing script that identifies the type of query and routes to the required models.
    """
    responses = {}
    
    # Text Modality: Sent to Model 1
    if modality == 'text':
        responses['Model 1'] = mock_model_1(input_data)
        
    # Audio Modality: Sent to Model 1 and Model 2
    elif modality == 'audio':
        responses['Model 1'] = mock_model_1(input_data)
        responses['Model 2'] = mock_model_2(input_data)
        
    # Video Modality: Sent to Model 1, Model 2, and Model 3
    elif modality == 'video':
        responses['Model 1'] = mock_model_1(input_data)
        responses['Model 2'] = mock_model_2(input_data)
        responses['Model 3'] = mock_model_3(input_data)
        
    return responses

def mock_model_1(data):
    return "I'm Model 1 and I received your query."

def mock_model_2(data):
    return "I'm Model 2 and I received your query."

def mock_model_3(data):
    return "I'm Model 3 and I received your query."

@app.route('/api/process', methods=['POST'])
def process_data():
    query_type = request.form.get('type')
    
    if query_type == 'text':
        text_content = request.form.get('text', '')
        if not text_content:
            return jsonify({"error": "Empty text query"}), 400
            
        result = preprocess_and_route(text_content, 'text')
        return jsonify({"status": "success", "modality": "Text", "results": result})
        
    elif query_type == 'file':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Guess Modality from mimetype
        mimetype, _ = mimetypes.guess_type(file.filename)
        if mimetype:
            if mimetype.startswith('audio'):
                result = preprocess_and_route(file.filename, 'audio')
                return jsonify({"status": "success", "modality": "Audio", "results": result})
            elif mimetype.startswith('video'):
                result = preprocess_and_route(file.filename, 'video')
                return jsonify({"status": "success", "modality": "Video", "results": result})
        
        return jsonify({"error": "Unsupported or unrecognized file format. Please upload a valid audio or video format."}), 400
        
    return jsonify({"error": "Invalid query type"}), 400

if __name__ == '__main__':
    print("Starting Model Router Backend...")
    print("Listening on http://localhost:5000")
    app.run(debug=True, port=5000)
