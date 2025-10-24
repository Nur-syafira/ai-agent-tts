#!/usr/bin/env python3

"""
Проверка доступности всех моделей и весов.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv

load_dotenv()


def check_faster_whisper():
    """Проверка faster-whisper."""
    print("📥 Проверка faster-whisper large-v3-turbo...")
    try:
        from faster_whisper import WhisperModel
        
        # Пытаемся загрузить модель (скачается если нет)
        print("   Загрузка модели (может занять время при первом запуске)...")
        model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="int8_float16",
        )
        
        print("   ✅ faster-whisper готов")
        print(f"      Device: cuda")
        print(f"      Model: large-v3-turbo")
        del model
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def check_silero_vad():
    """Проверка Silero VAD."""
    print("\n📥 Проверка Silero VAD...")
    try:
        import torch
        
        print("   Загрузка модели...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
        )
        
        print("   ✅ Silero VAD готов")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def check_kokoro():
    """Проверка Kokoro-82M."""
    print("\n📥 Проверка Kokoro-82M...")
    try:
        from kokoro import KPipeline
        
        print("   Инициализация pipeline...")
        pipeline = KPipeline(lang_code='a')
        
        print("   ✅ Kokoro-82M готов")
        print(f"      Voices: 9 (af_heart, af_bella, af_sarah, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis)")
        return True
        
    except ImportError:
        print("   ❌ Kokoro не установлен")
        print("      Установи: pip install kokoro>=0.9.2 misaki[en]")
        return False
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def check_piper():
    """Проверка Piper TTS."""
    print("\n📥 Проверка Piper TTS...")
    
    # Проверяем бинарник piper
    import subprocess
    try:
        result = subprocess.run(
            ["piper", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        print(f"   ✅ Piper установлен: {result.stdout.strip()}")
    except FileNotFoundError:
        print("   ⚠️  Piper не установлен (опционально)")
        print("      Установи: wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz")
        return False
    except Exception as e:
        print(f"   ⚠️  Piper проверка не удалась: {e}")
        return False
    
    # Проверяем модель
    model_path = Path("models/ru_RU-dmitri-medium.onnx")
    if model_path.exists():
        print(f"   ✅ Русская модель найдена: {model_path}")
        return True
    else:
        print(f"   ⚠️  Русская модель не найдена: {model_path}")
        print("      Скачай через: ./venv/bin/python scripts/download_models.py")
        return False


def check_qwen_availability():
    """Проверка доступности Qwen2.5-14B-Instruct-AWQ."""
    print("\n📥 Проверка Qwen2.5-14B-Instruct-AWQ...")
    
    # Проверяем в HuggingFace cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    model_name = "models--Qwen--Qwen2.5-14B-Instruct-AWQ"
    model_cache = cache_dir / model_name
    
    if model_cache.exists():
        print(f"   ✅ Модель найдена в кэше: {model_cache}")
        size_gb = sum(f.stat().st_size for f in model_cache.rglob('*') if f.is_file()) / (1024**3)
        print(f"      Размер: {size_gb:.2f} GB")
        return True
    else:
        print(f"   ⚠️  Модель не найдена в кэше")
        print("      Скачается автоматически при запуске vLLM (первый запуск ~15-20 минут)")
        return False


def check_cuda():
    """Проверка CUDA."""
    print("\n🔍 Проверка CUDA...")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA доступна")
            print(f"      Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"      GPU {i}: {props.name}")
                print(f"         VRAM: {props.total_memory / (1024**3):.2f} GB")
            return True
        else:
            print("   ❌ CUDA недоступна")
            return False
            
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def main():
    """Главная функция."""
    print("=" * 70)
    print(" " * 20 + "Sales Agent - Model Checker")
    print("=" * 70)
    print()
    
    results = {}
    
    # Проверка CUDA
    results["cuda"] = check_cuda()
    
    # Проверка моделей
    results["faster_whisper"] = check_faster_whisper()
    results["silero_vad"] = check_silero_vad()
    results["kokoro"] = check_kokoro()
    results["piper"] = check_piper()
    results["qwen"] = check_qwen_availability()
    
    # Итоговый статус
    print()
    print("=" * 70)
    print("Итоговый статус:")
    print("=" * 70)
    
    for name, status in results.items():
        emoji = "✅" if status else "⚠️"
        print(f"  {emoji} {name:.<30} {'OK' if status else 'Not ready'}")
    
    print()
    
    critical = ["cuda", "faster_whisper", "kokoro"]
    if all(results.get(k, False) for k in critical):
        print("✅ Все критичные компоненты готовы к работе!")
        print()
        print("Можешь запускать сервисы:")
        print("  1. vLLM: vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ ...")
        print("  2. ASR Gateway: ./venv/bin/python src/asr_gateway/main.py")
        print("  3. TTS Gateway: ./venv/bin/python src/tts_gateway/main.py")
        print("  4. Policy Engine: ./venv/bin/python src/policy_engine/main.py")
    else:
        print("⚠️  Некоторые компоненты не готовы.")
        print()
        print("Установи недостающие компоненты и повтори проверку.")


if __name__ == "__main__":
    main()

