uv run tools/tokenizer/extend_bpe.py `
    --base-model checkpoints/bpe.model `
    --manifests datasets/JA_emilia/ja_emilia.json `
    --output-model checkpoints/extended_bpe.model `
    --vocab-size 24000 `
    --character-coverage 1.0 `
    --model-type bpe