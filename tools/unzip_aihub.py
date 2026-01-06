import zipfile
import os
from pathlib import Path
from tqdm import tqdm

def unzip_all(src_dir, dest_dir):
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = list(src_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files in {src_dir}")
    
    for zip_path in tqdm(zip_files, desc=f"Unzipping {src_dir.name}"):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

if __name__ == "__main__":
    base_path = Path(r"g:\audiolm\index-tts\datasets\KR\133.감성 및 발화 스타일 동시 고려 음성합성 데이터\01-1.정식개방데이터")
    extract_base = Path(r"g:\audiolm\index-tts\datasets\KR\extracted")
    
    configs = [
        (base_path / "Training" / "01.원천데이터", extract_base / "Training" / "audio"),
        (base_path / "Training" / "02.라벨링데이터", extract_base / "Training" / "label"),
        (base_path / "Validation" / "01.원천데이터", extract_base / "Validation" / "audio"),
        (base_path / "Validation" / "02.라벨링데이터", extract_base / "Validation" / "label"),
    ]
    
    for src, dest in configs:
        if src.exists():
            unzip_all(src, dest)
        else:
            print(f"Directory not found: {src}")
