import torch
import sys
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.config import config
from src.models.affective_encoder import AffectiveEncoder
from src.data.dataset import AudioEmotionDataset, collate_fn
from tqdm import tqdm

def evaluate(model, dataloader, device):
    model.eval()
    correct_categorical = 0
    total_samples = 0
    val_error = 0
    arousal_error = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Evaluation Pipeline on {device}...")
    
    # 1. Load Model Architecture & Weights
    model = AffectiveEncoder(config)
    model_weight_path = Path("aer_base_model_epoch_9.pt")
    
    if not model_weight_path.exists():
        print(f"Error: Could not find trained weights at {model_weight_path}")
        sys.exit(1)
        
    print(f"Loading weights from {model_weight_path}...")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    
    # 2. Load Dataset Manifest
    manifest_path = config.processed_dir / "train_manifest.csv"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)
        
    df = pd.read_csv(manifest_path)
    
    # 3. Sample exactly 10% of the data for testing
    sample_fraction = 0.10
    test_df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
    print(f"Randomly sampled {len(test_df)} audio files ({sample_fraction*100}% of total dataset) for testing.")
    
    # 4. Initialize DataLoader
    test_dataset = AudioEmotionDataset(test_df, config)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size * 2, # Can double batch size during eval since no gradients
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 5. Execute
    evaluate(model, test_loader, device)
