uv run python inference_script.py ^
    --config checkpoints/config_finetune.yaml ^
    --gpt-checkpoint trained_ckpts_ko/model_step5220.pth ^
    --speaker "examples/06.wav" ^
    --text-file "test_text.txt" ^
    --output "outputs/test_result_ko.wav" ^
    --device cuda:0

@rem === 감정 적용 테스트 예시 (위 명령어를 주석 처리하고 아래 중 하나를 사용해 보세요) ===

@rem 1. 텍스트 기반 감정 추론 (Qwen-Emotion 사용)
@rem uv run python inference_script.py ^
@rem     --config checkpoints/config_finetune.yaml ^
@rem     --gpt-checkpoint trained_ckpts_ko/model_step5220.pth ^
@rem     --speaker "datasets/KR/extracted/Training/audio/A-A1-C-001-0101.wav" ^
@rem     --text-file "test_text.txt" ^
@rem     --use-emo-text ^
@rem     --emo-text "정말 화가 나는 상황이야!" ^
@rem     --output "outputs/test_result_ko_angry.wav" ^
@rem     --device cuda:0

@rem 2. 외부 오디오 기반 감정 복제
@rem uv run python inference_script.py ^
@rem     --config checkpoints/config_finetune.yaml ^
@rem     --gpt-checkpoint trained_ckpts_ko/model_step5220.pth ^
@rem     --speaker "datasets/KR/extracted/Training/audio/A-A1-C-001-0101.wav" ^
@rem     --text-file "test_text.txt" ^
@rem     --emo-audio "path/to/emotional_audio.wav" ^
@rem     --emo-alpha 1.0 ^
@rem     --output "outputs/test_result_ko_ext_emo.wav" ^
@rem     --device cuda:0
