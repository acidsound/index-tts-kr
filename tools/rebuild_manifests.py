import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

def rebuild_manifest(source_manifest, output_manifest, data_dir):
    data_dir = Path(data_dir)
    print(f"Rebuilding {output_manifest} from {source_manifest}...")
    
    records = []
    with open(source_manifest, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            
    with open(output_manifest, "w", encoding="utf-8") as f:
        for record in tqdm(records):
            uid = record["id"]
            codes_path = data_dir / "codes" / f"{uid}.npy"
            text_path = data_dir / "text_ids" / f"{uid}.npy"
            cond_path = data_dir / "condition" / f"{uid}.npy"
            emo_path = data_dir / "emo_vec" / f"{uid}.npy"
            
            if not codes_path.exists() or not text_path.exists():
                continue
                
            # Load lengths
            code_len = np.load(codes_path).shape[0]
            text_len = np.load(text_path).shape[0]
            condition_len = np.load(cond_path).shape[0]
            
            entry = {
                "id": uid,
                "audio_path": record["audio"],
                "text": record["text"],
                "speaker": record["speaker"],
                "language": "ko",
                "text_ids_path": f"text_ids/{uid}.npy",
                "text_len": int(text_len),
                "codes_path": f"codes/{uid}.npy",
                "code_len": int(code_len),
                "condition_path": f"condition/{uid}.npy",
                "condition_len": int(condition_len),
                "emo_vec_path": f"emo_vec/{uid}.npy"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    rebuild_manifest("datasets/KR/train.jsonl", "ko_processed_data/train_manifest.jsonl", "ko_processed_data")
    rebuild_manifest("datasets/KR/val.jsonl", "ko_processed_data/val_manifest.jsonl", "ko_processed_data")
