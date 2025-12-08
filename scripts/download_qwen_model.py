#!/usr/bin/env python3

"""
Скрипт для загрузки модели Qwen3-16B-A3B-abliterated-AWQ в локальную директорию.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()


def download_qwen_model():
    """Скачивает Qwen3-16B-A3B-abliterated-AWQ в локальную директорию."""
    print("=" * 70)
    print(" " * 15 + "Qwen3-16B-A3B-abliterated-AWQ Downloader")
    print("=" * 70)
    print()
    
    model_name = "warshanks/Qwen3-16B-A3B-abliterated-AWQ"
    target_dir = Path(__file__).parent.parent / "models" / "Qwen3-16B-A3B-abliterated-AWQ"
    
    # Создаём директорию если её нет
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading {model_name}...")
    print(f"   Target directory: {target_dir}")
    print()
    print("   This may take 10-30 minutes depending on your internet speed.")
    print("   Model size: ~6 GB (AWQ quantized)")
    print()
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,  # Копируем файлы, не симлинки
            resume_download=True,
        )
        
        print()
        print("=" * 70)
        print("✅ Model downloaded successfully!")
        print("=" * 70)
        print()
        print(f"Model location: {target_dir}")
        print()
        print("Next steps:")
        print("1. Update src/llm_service/config.yaml:")
        print("   model:")
        print('     name: "models/Qwen3-16B-A3B-abliterated-AWQ"')
        print()
        print("2. Start vLLM server:")
        print(f'   vllm serve {target_dir} \\')
        print("     --host 0.0.0.0 --port 8000 \\")
        print("     --max-model-len 2048 --gpu-memory-utilization 0.75 \\")
        print("     --quantization awq --enable-chunked-prefill --enable-prefix-caching")
        print()
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Error downloading model: {e}")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print("- Check your internet connection")
        print("- Ensure you have enough disk space (~10 GB free)")
        print("- Try running: huggingface-cli login")
        print()
        raise


if __name__ == "__main__":
    download_qwen_model()

