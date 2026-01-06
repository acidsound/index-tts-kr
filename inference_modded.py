#!/usr/bin/env python3

"""
Convenience script for running IndexTTS2 MODDED inference from the command line.
This script uses indextts/infer_v2_modded.py which supports:
- Automatic vocab resizing (no more strict config/checkpoint mismatch)
- Duration control via --duration
- Efficient reference audio caching
- Improved silence removal
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

# Use the modded version for advanced features
from indextts.infer_v2_modded import IndexTTS2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IndexTTS2 (MODDED version) inference with advanced controls."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to the YAML config that defines checkpoint/tokenizer paths (default: checkpoints/config.yaml).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="checkpoints",
        help="Directory containing weights/tokenizer referenced in the config (default: checkpoints).",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        required=True,
        help="Reference speaker audio (wav/mp3/etc.).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        type=str,
        help="Text to synthesise.",
    )
    group.add_argument(
        "--text-file",
        type=str,
        help="Path to a UTF-8 text file containing the sentence to synthesise.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output wav path (default: output.wav).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 for GPT inference.",
    )
    parser.add_argument(
        "--gpt-checkpoint",
        type=str,
        default=None,
        help="Override the GPT checkpoint path provided in the config.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Override the BPE tokenizer path provided in the config.",
    )
    parser.add_argument(
        "--emo-audio",
        type=str,
        default=None,
        help="Optional emotion reference audio clip.",
    )
    parser.add_argument(
        "--emo-alpha",
        type=float,
        default=1.0,
        help="Blend factor for the emotion reference audio (default: 1.0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Target duration in seconds (MODDED version only).",
    )
    parser.add_argument(
        "--use-accel",
        action="store_true",
        help="Enable acceleration engine (requires flash-attn).",
    )
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=120,
        help="Maximum text tokens per segment (default: 120).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Sampling top-p (default: 0.8).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    parser.add_argument(
        "--interval-silence",
        type=int,
        default=200,
        help="Silence (ms) to insert between segments (default: 200).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging.",
    )

    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    text_path = Path(args.text_file)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    return text_path.read_text(encoding="utf-8").strip()


def build_generation_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "top_p": args.top_p,
        "temperature": args.temperature,
    }
    return kwargs


def main() -> None:
    args = parse_args()
    text = load_text(args)
    generation_kwargs = build_generation_kwargs(args)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)
    model_dir_resolved = Path(args.model_dir).resolve()

    # Apply overrides
    gpt_path = args.gpt_checkpoint if args.gpt_checkpoint else None
    tokenizer_path = args.tokenizer if args.tokenizer else None

    # Modded version handles these as direct args
    engine = IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(model_dir_resolved),
        device=args.device,
        use_fp16=args.fp16,
        use_accel=args.use_accel,
        gpt_checkpoint_path=gpt_path,
        bpe_model_path=tokenizer_path,
    )

    engine.infer(
        spk_audio_prompt=args.speaker,
        text=text,
        output_path=args.output,
        emo_audio_prompt=args.emo_audio,
        emo_alpha=args.emo_alpha,
        duration_seconds=args.duration,
        interval_silence=args.interval_silence,
        verbose=args.verbose,
        max_text_tokens_per_sentence=args.max_text_tokens,
        **generation_kwargs,
    )
    
    print(f"\n>> [DONE] Output saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
