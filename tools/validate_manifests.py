from pathlib import Path
import json

def validate_jsonl(p):
    print(f"Checking {p}...")
    content = p.read_text(encoding="utf-8")
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except Exception as e:
            print(f"Line {i+1} failed: {e}")
            print(f"Content length: {len(line)}")
            # Show 10 chars around the error index if possible
            # Column usually given in e.pos for JSONDecodeError
            if hasattr(e, 'pos'):
                start = max(0, e.pos - 20)
                end = min(len(line), e.pos + 20)
                print(f"Context: {line[start:end]!r}")
                print(f"Error at char {e.pos}: {line[e.pos]!r}")

if __name__ == "__main__":
    validate_jsonl(Path("ko_processed_data/train_manifest.jsonl"))
    validate_jsonl(Path("ko_processed_data/val_manifest.jsonl"))
