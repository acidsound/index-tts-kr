import json
from pathlib import Path
import sys
import os

# Add project root to sys.path so we can import indextts
sys.path.append(os.getcwd())

from indextts.utils.common import tokenize_by_CJK_char
from tqdm import tqdm

def prepare_corpus(manifest_paths, output_path):
    print(f"Reading manifests: {manifest_paths}")
    total = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for manifest_path in manifest_paths:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Processing {manifest_path.name}"):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = record.get("text", "")
                    if not text:
                        continue
                    
                    # Apply CJK char tokenization (adds spaces between characters)
                    # This matches the behavior of TextTokenizer.tokenize()
                    tokenized_text = tokenize_by_CJK_char(text)
                    out_f.write(tokenized_text + "\n")
                    total += 1
    print(f"Done! Wrote {total} pre-tokenized lines to {output_path}")

if __name__ == "__main__":
    manifests = [
        Path("datasets/KR/train.jsonl"),
        Path("datasets/KR/val.jsonl")
    ]
    output = Path("datasets/KR/ko_corpus_pretokenized.txt")
    prepare_corpus(manifests, output)
