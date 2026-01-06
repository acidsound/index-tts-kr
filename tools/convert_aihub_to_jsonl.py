import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_aihub(base_dir, output_file, mode="Training"):
    label_dir = base_dir / mode / "label"
    audio_dir = base_dir / mode / "audio"
    
    json_files = list(label_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {label_dir}")
    
    records = []
    skipped = 0
    
    for json_path in tqdm(json_files, desc=f"Converting {mode}"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # AI Hub JSON can be a single dict or a list of dicts
            if isinstance(data, dict):
                data_list = [data]
            else:
                data_list = data
            
            for item in data_list:
                # Use filename part 2 (e.g., A101, H102) as speaker ID for better diversity
                stem = json_path.stem
                parts = stem.split("-")
                if len(parts) >= 2:
                    speaker_id = parts[1]
                else:
                    speaker_id = str(item.get("reciter", {}).get("id", "unknown"))
                
                for sentence in item.get("sentences", []):
                    voice_piece = sentence.get("voice_piece", {})
                    filename = voice_piece.get("filename")
                    text = voice_piece.get("tr")
                    uid = sentence.get("id")
                    
                    if not filename or not text or not uid:
                        skipped += 1
                        continue
                    
                    # Pre-tokenize text for BPE compatibility
                    from indextts.utils.common import tokenize_by_CJK_char
                    text = tokenize_by_CJK_char(text)
                    
                    audio_path = audio_dir / filename
                    if not audio_path.exists():
                        skipped += 1
                        continue
                    
                    rel_audio_path = f"datasets/KR/extracted/{mode}/audio/{filename}"
                    
                    records.append({
                        "id": uid,
                        "audio": rel_audio_path,
                        "text": text,
                        "speaker": f"aihub_{speaker_id}",
                        "language": "ko"
                    })
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            skipped += 1
            
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Done! Saved {len(records)} records to {output_file}. Skipped {skipped}.")

if __name__ == "__main__":
    base_path = Path("datasets/KR/extracted")
    
    # Training
    convert_aihub(base_path, Path("datasets/KR/train.jsonl"), mode="Training")
    
    # Validation
    convert_aihub(base_path, Path("datasets/KR/val.jsonl"), mode="Validation")
