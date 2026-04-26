from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mimetypes
import os
import sys
import torch
from pathlib import Path

# Add AER_Encoder to python path for module loading
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "AER_Encoder"))

try:
    from src.config.config import config
    from src.models.affective_encoder import AffectiveEncoder
    from scripts.inference import run_aer_inference
except ImportError as e:
    print(f"Warning: AER Module not found or failed to load. {e}")

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS so the separate frontend can communicate with the backend

# Initialize AER Model globally on server spin-up
print("Initializing AER_Encoder globally...")
try:
    device = torch.device("cpu") # Router inference runs on CPU by default
    aer_model = AffectiveEncoder(config)
    
    # Target the newest trained checkpoint inside AER_Encoder dir
    model_weight_path = BASE_DIR / "AER_Encoder" / "aer_base_model_epoch_9.pt"
    if model_weight_path.exists():
        aer_model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print("-> AER_Encoder trained weights loaded successfully.")
    else:
        print(f"-> WARNING: Weights not found at {model_weight_path}. AER is running an untrained shell!")
    aer_model.eval()
except Exception as e:
    print(f"Failed to load AER_Encoder: {e}")
    aer_model = None

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
        
    # Audio Modality: Sent to Model 1 and AER_Encoder
    elif modality == 'audio':
        responses['Model 1'] = mock_model_1(input_data)
        responses['AER_Encoder'] = analyze_audio_aer(input_data)
        
    # Video Modality: Sent to Model 1, AER_Encoder, and Model 3
    elif modality == 'video':
        responses['Model 1'] = mock_model_1(input_data)
        responses['AER_Encoder'] = analyze_audio_aer(input_data)
        responses['Model 3'] = mock_model_3(input_data)
        
    return responses

def mock_model_1(data):
    return "I'm Model 1 and I received your query."

def analyze_audio_aer(file_path):
    """
    Passes the cached audio file directly into the PyTorch AER Neural Network.
    Returns the JSON payload containing predicted emotion, valence, arousal, and latents.
    """
    if aer_model is None:
        return {"error": "AER_Encoder backend is offline."}
    
    try:
        # Run inference using the imported logic from scripts/inference.py
        payload = run_aer_inference(aer_model, file_path)
        return payload
    except Exception as e:
        return {"error": f"AER Inference Failed: {str(e)}"}

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
            
        # Securely cache the file to disk so PyTorch/torchaudio can mathematically decode it
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        file.save(str(file_path))
            
        # Guess Modality from mimetype
        mimetype, _ = mimetypes.guess_type(file.filename)
        if mimetype:
            if mimetype.startswith('audio'):
                result = preprocess_and_route(str(file_path), 'audio')
                return jsonify({"status": "success", "modality": "Audio", "results": result})
            elif mimetype.startswith('video'):
                result = preprocess_and_route(str(file_path), 'video')
                return jsonify({"status": "success", "modality": "Video", "results": result})
        
        return jsonify({"error": "Unsupported or unrecognized file format. Please upload a valid audio or video format."}), 400
        
    return jsonify({"error": "Invalid query type"}), 400

if __name__ == '__main__':
    print("Starting Model Router Backend...")
    print("Listening on http://localhost:5000")
    app.run(debug=True, port=5000)
