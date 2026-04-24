import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Ensure config loads
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.config import config

# The 7-Class Unified Taxonomy
# 0: Neutral, 1: Happy, 2: Sad, 3: Angry, 4: Fear, 5: Surprise, 6: Disgust
UNIFIED_EMOTION_MAP = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fear': 4,
    'surprise': 5,
    'disgust': 6
}

# Optional: Valence / Arousal mapping approximations per class (Scale: -1.0 to 1.0)
V_A_APPROXIMATION = {
    0: (0.0, 0.0),    # Neutral
    1: (0.8, 0.6),    # Happy (Positive, High Arousal)
    2: (-0.8, -0.6),  # Sad (Negative, Low Arousal)
    3: (-0.8, 0.8),   # Angry (Negative, High Arousal)
    4: (-0.6, 0.9),   # Fear (Negative, Very High Arousal)
    5: (0.4, 0.8),    # Surprise (Positive/Neutral, High Arousal)
    6: (-0.9, 0.4)    # Disgust (Very Negative, Moderate Arousal)
}

def parse_ravdess(base_dir):
    records = []
    # Format: 03-01-05-01-01-01-01.wav -> Emotion is the 3rd token (05=angry)
    # RAVDESS map: 01=neutral, 02=calm(map->neutral), 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise
    rav_map = {'01':'neutral', '02':'neutral', '03':'happy', '04':'sad', '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'}
    
    files = glob.glob(str(base_dir / "ravdess" / "**" / "*.wav"), recursive=True)
    for f in files:
        basename = os.path.basename(f).split('.')[0]
        parts = basename.split('-')
        if len(parts) >= 3:
            emo_code = parts[2]
            if emo_code in rav_map:
                records.append({
                    'file_path': f,
                    'dataset': 'RAVDESS',
                    'emotion': rav_map[emo_code],
                    'label_idx': UNIFIED_EMOTION_MAP[rav_map[emo_code]]
                })
    return records

def parse_tess(base_dir):
    records = []
    # Format: folder determines emotion (e.g. OAF_angry, YAF_happy)
    # Suffix is also _angry, _happy, _ps (pleasant surprise), _fear, _sad, _disgust, _neutral
    files = glob.glob(str(base_dir / "tess" / "**" / "*.wav"), recursive=True)
    for f in files:
        basename = os.path.basename(f).lower()
        if 'neutral' in basename: emo = 'neutral'
        elif 'happy' in basename: emo = 'happy'
        elif 'sad' in basename: emo = 'sad'
        elif 'angry' in basename: emo = 'angry'
        elif 'fear' in basename: emo = 'fear'
        elif 'disgust' in basename: emo = 'disgust'
        elif 'ps' in basename or 'surprise' in basename: emo = 'surprise'
        else: continue
            
        records.append({
            'file_path': f,
            'dataset': 'TESS',
            'emotion': emo,
            'label_idx': UNIFIED_EMOTION_MAP[emo]
        })
    return records

def parse_crema(base_dir):
    records = []
    # Format: 1001_DFA_ANG_XX.wav -> 3rd token is emotion
    # CREMA map: NEU, HAP, SAD, ANG, FEA, DIS
    crema_map = {'NEU':'neutral', 'HAP':'happy', 'SAD':'sad', 'ANG':'angry', 'FEA':'fear', 'DIS':'disgust'}
    
    files = glob.glob(str(base_dir / "crema" / "**" / "*.wav"), recursive=True)
    for f in files:
        basename = os.path.basename(f).split('.')[0]
        parts = basename.split('_')
        if len(parts) >= 3:
            emo_code = parts[2]
            if emo_code in crema_map:
                records.append({
                    'file_path': f,
                    'dataset': 'CREMA',
                    'emotion': crema_map[emo_code],
                    'label_idx': UNIFIED_EMOTION_MAP[crema_map[emo_code]]
                })
    return records

def parse_savee(base_dir):
    records = []
    # Format: DC_a01.wav or JE_sa01.wav -> first letters before digits is emotion
    # SAVEE map: a=angry, d=disgust, f=fear, h=happy, n=neutral, sa=sad, su=surprise
    savee_map = {'a':'angry', 'd':'disgust', 'f':'fear', 'h':'happy', 'n':'neutral', 'sa':'sad', 'su':'surprise'}
    
    files = glob.glob(str(base_dir / "SAVEE" / "**" / "*.wav"), recursive=True)
    for f in files:
        basename = os.path.basename(f).split('_')[-1] # sa01.wav -> sa01.wav
        
        # strip numbers and .wav
        import re
        emo_code = re.sub(r'[0-9]+', '', basename.split('.')[0]).lower()
        
        if emo_code in savee_map:
            records.append({
                'file_path': f,
                'dataset': 'SAVEE',
                'emotion': savee_map[emo_code],
                'label_idx': UNIFIED_EMOTION_MAP[savee_map[emo_code]]
            })
    return records

def generate_distribution_chart(df, save_path):
    plt.figure(figsize=(10, 6))
    
    # Custom elegant color palette ensuring aesthetics
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x='emotion', hue='dataset', data=df, palette='viridis')
    
    plt.title('Audio Samples per Emotion Class across Datasets (System Config: 7-Class)', fontsize=14, pad=15)
    plt.xlabel('Emotion Classification', fontsize=12)
    plt.ylabel('Number of Audio Samples', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Sub-Dataset Origin', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"-> Saved distribution chart to {save_path}")

def main():
    print("Initiating Dataset Standardizer (M1 AER Pipeline)...")
    
    base_raw_dir = config.raw_generic_dir
    if not base_raw_dir.exists():
        print(f"Error: Directory {base_raw_dir} does not exist.")
        return

    # Aggregate records
    all_records = []
    all_records.extend(parse_ravdess(base_raw_dir))
    all_records.extend(parse_tess(base_raw_dir))
    all_records.extend(parse_crema(base_raw_dir))
    all_records.extend(parse_savee(base_raw_dir))
    
    df = pd.DataFrame(all_records)
    
    if len(df) == 0:
        print("No audio files found! Check directory structures inside data/raw/generic/")
        return
        
    print(f"Successfully aggregated {len(df)} total audio samples across {df['dataset'].nunique()} datasets.")
    
    # Enrich with approximated Valence/Arousal values
    df['valence'] = df['label_idx'].apply(lambda idx: V_A_APPROXIMATION[idx][0])
    df['arousal'] = df['label_idx'].apply(lambda idx: V_A_APPROXIMATION[idx][1])
    
    # Save unified CSV
    manifest_path = config.processed_dir / "train_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"-> Saved universal training manifest to {manifest_path}")
    
    # Generate Chart
    chart_path = config.processed_dir / "class_distribution.png"
    generate_distribution_chart(df, chart_path)
    
    print("\nClass Breakdown summary:")
    print(df['emotion'].value_counts())
    print("\nDataset fully synchronized and standardized to the 7-Class Unified Taxonomy.")

if __name__ == "__main__":
    main()
