@echo off

@rem == IndexTTS2 Modded Inference Script (Korean) ==
@rem This script uses the modded inference class which supports duration control and faster caching.

uv run python inference_modded.py ^
    --config checkpoints/config_finetune.yaml ^
    --gpt-checkpoint trained_ckpts_ko/model_step5584.pth ^
    --speaker "outputs/2002_cn.ogg" ^
    --text-file "test_text.txt" ^
    --output "outputs/test_result_modded.wav" ^
    --duration 10.0 ^
    --use-accel ^
    --fp16 ^
    --temperature 0.8 ^
    --top-p 0.9 ^
    --device cuda:0

echo.
echo Inference complete.
