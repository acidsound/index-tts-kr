@echo off
setlocal

:: WebUI 실행을 위한 한국어 모델용 배치 파일
uv run webui.py ^
    --config checkpoints/config_finetune.yaml ^
    --gpt_checkpoint trained_ckpts_ko/model_step5584.pth ^
    --bpe_model checkpoints/bpe_multilingual.model ^
    --is_fp16

endlocal
