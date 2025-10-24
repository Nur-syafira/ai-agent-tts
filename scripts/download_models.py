#!/usr/bin/env python3

"""
Скрипт для скачивания моделей (ASR, LLM, TTS).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import subprocess
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()


def download_whisper_model():
    """Скачивает faster-whisper модель (large-v3-turbo)."""
    print("📥 Downloading faster-whisper large-v3-turbo...")
    
    # faster-whisper скачивается автоматически при первом использовании
    # Но можно предзагрузить через CT2:
    try:
        from faster_whisper import WhisperModel
        
        model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="int8_float16",
        )
        
        print("✅ faster-whisper model ready")
        del model
        
    except Exception as e:
        print(f"⚠️  Failed to download faster-whisper: {e}")


def download_qwen_model():
    """Скачивает Qwen2.5-14B-Instruct-AWQ."""
    print("📥 Downloading Qwen2.5-14B-Instruct-AWQ...")
    
    model_name = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    try:
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
        )
        
        print(f"✅ {model_name} downloaded")
        
    except Exception as e:
        print(f"⚠️  Failed to download Qwen: {e}")
        print("   Model will be downloaded automatically when vLLM starts")


def download_piper_model():
    """Скачивает Piper TTS модель (русский)."""
    print("📥 Downloading Piper TTS model (ru_RU-dmitri-medium)...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx"
    config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json"
    
    model_path = models_dir / "ru_RU-dmitri-medium.onnx"
    config_path = models_dir / "ru_RU-dmitri-medium.onnx.json"
    
    if model_path.exists():
        print(f"   Model already exists: {model_path}")
    else:
        subprocess.run(["wget", "-O", str(model_path), model_url], check=True)
        print(f"✅ Downloaded: {model_path}")
    
    if config_path.exists():
        print(f"   Config already exists: {config_path}")
    else:
        subprocess.run(["wget", "-O", str(config_path), config_url], check=True)
        print(f"✅ Downloaded: {config_path}")


def download_silero_vad():
    """Скачивает Silero VAD модель."""
    print("📥 Downloading Silero VAD...")
    
    try:
        import torch
        
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True,
        )
        
        print("✅ Silero VAD model ready")
        
    except Exception as e:
        print(f"⚠️  Failed to download Silero VAD: {e}")


def main():
    """Главная функция."""
    print("=" * 60)
    print("Sales Agent - Model Downloader")
    print("=" * 60)
    print()
    
    # Создаём папку для моделей
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Скачиваем модели
    download_whisper_model()
    print()
    
    download_qwen_model()
    print()
    
    download_piper_model()
    print()
    
    download_silero_vad()
    print()
    
    print("=" * 60)
    print("✅ All models downloaded!")
    print("=" * 60)
    print()
    print("Note: Some models download automatically on first use.")
    print("      If you see warnings, it's okay - they'll download when needed.")


if __name__ == "__main__":
    main()

