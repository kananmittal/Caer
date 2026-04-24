import torch
import torchaudio
import json
import sys
from pathlib import Path

# Setup paths
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.config import config
from src.models.affective_encoder import AffectiveEncoder

EMOTION_MAP = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fear"
}

def load_and_preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != config.target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.target_sample_rate)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    return waveform.squeeze(0).unsqueeze(0) # [1, Seq_Len]

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_audio_file>")
        sys.exit(1)
        
    target_audio = sys.argv[1]
    
    device = torch.device("cpu") # For inference testing
    
    print("Loading AER Model Architecture...")
    model = AffectiveEncoder(config)
    model.eval()
    # Note: In production you would load your trained weights here:
    # model.load_state_dict(torch.load("aer_weights.pt"))
    
    print(f"Ingesting audio: {target_audio}")
    input_values = load_and_preprocess_audio(target_audio)
    
    print("Encoding...")
    with torch.no_grad():
        outputs = model(input_values)
        
    # Process outputs
    pred_class_idx = torch.argmax(outputs["categorical_logits"], dim=1).item()
    emotion_label = EMOTION_MAP.get(pred_class_idx, "Unknown")
    
    valence_score = outputs["valence"].item()
    arousal_score = outputs["arousal"].item()
    
    latent_vector = outputs["latent_vector"].squeeze(0).tolist()
    
    response_payload = {
        "status": "success",
        "model": "AER_Encoder",
        "results": {
            "predicted_emotion": emotion_label,
            "valence": round(valence_score, 4),
            "arousal": round(arousal_score, 4),
            "latent_vector_dimension": len(latent_vector),
            "latent_vector_preview": latent_vector[:5] # Showing first 5 for brevity
        }
    }
    
    print("\n--- INFERENCE RESULTS ---")
    print(json.dumps(response_payload, indent=4))
    print("-------------------------\n")
    print("Latent Vector successfully extracted. Ready for Vector Database ingestion!")

if __name__ == "__main__":
    main()
