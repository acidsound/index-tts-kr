from pathlib import Path
import json
from collections import Counter

def check_speakers(base_dir):
    speakers_reciter = Counter()
    speakers_file_p2 = Counter()
    speakers_file_p3 = Counter()
    
    json_files = list(base_dir.glob("**/*.json"))
    print(f"Checking {len(json_files)} JSON files...")
    
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                data_list = [data]
            else:
                data_list = data
            
            for item in data_list:
                reciter_id = item.get("reciter", {}).get("id")
                if reciter_id:
                    speakers_reciter[reciter_id] += 1
                
                stem = json_path.stem
                parts = stem.split("-")
                if len(parts) >= 3:
                     speakers_file_p2[parts[1]] += 1
                     speakers_file_p3[parts[2]] += 1
                
        except Exception:
            pass
    
    print(f"\nUnique 'reciter.id' values: {len(speakers_reciter)}")
    print(f"Unique filename part 2 values: {len(speakers_file_p2)}")
    print(f"Unique filename part 3 values: {len(speakers_file_p3)}")
    
    print("\nTop reciter IDs:", speakers_reciter.most_common(5))
    print("Top Filename Part 2:", speakers_file_p2.most_common(5))
    print("Top Filename Part 3:", speakers_file_p3.most_common(5))

if __name__ == "__main__":
    check_speakers(Path("datasets/KR/extracted"))
