import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.config import config
from src.models.affective_encoder import AffectiveEncoder

def evaluate(model, dataloader, device):
    model.eval()
    correct_categorical = 0
    total_samples = 0
    val_error = 0
    arousal_error = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            cat_labels = batch["categorical_label"].to(device)
            valence_labels = batch["valence"].to(device)
            arousal_labels = batch["arousal"].to(device)
            
            outputs = model(input_values)
            
            # Accuracy computation for Categorical Label
            preds = torch.argmax(outputs["categorical_logits"], dim=1)
            correct_categorical += (preds == cat_labels).sum().item()
            total_samples += cat_labels.size(0)
            
            # Mean Absolute Error for Dimensional
            val_error += torch.abs(outputs["valence"] - valence_labels).sum().item()
            arousal_error += torch.abs(outputs["arousal"] - arousal_labels).sum().item()
            
    accuracy = correct_categorical / total_samples if total_samples > 0 else 0
    mae_val = val_error / total_samples if total_samples > 0 else 0
    mae_ar = arousal_error / total_samples if total_samples > 0 else 0
    
    print(f"Evaluation Complete!")
    print(f"Categorical Accuracy: {accuracy * 100:.2f}%")
    print(f"Valence MAE: {mae_val:.4f}")
    print(f"Arousal MAE: {mae_ar:.4f}")
    
    return accuracy, mae_val, mae_ar

if __name__ == "__main__":
    print("AER Evaluation Script Ready. Awaiting data loaders.")
