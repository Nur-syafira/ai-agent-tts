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
    print("📥 Проверка faster-whisper large-v3-turbo (Dropbox)...")
    try:
        from faster_whisper import WhisperModel
        
        # Пытаемся загрузить модель (скачается если нет)
        print("   Загрузка модели (может занять время при первом запуске)...")
        model = WhisperModel(
            "dropbox-dash/faster-whisper-large-v3-turbo",
            device="cuda",
            compute_type="int8_float16",
        )
        
        print("   ✅ faster-whisper готов")
        print(f"      Device: cuda")
        print(f"      Model: dropbox-dash/faster-whisper-large-v3-turbo")
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


def check_f5_tts():
    """Проверка F5-TTS."""
    print("\n📥 Проверка F5-TTS...")
    try:
        from src.tts_gateway.f5_tts_engine import F5TTSEngine
        import logging
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)  # Уменьшаем логирование
        
        print("   Инициализация F5-TTS (может занять время при первом запуске)...")
        # Используем локальный путь к модели
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "F5-tts"
        
        f5_tts = F5TTSEngine(
            model_path=str(model_path),
            device="cuda",
            sample_rate=24000,
            use_stress_marks=True,
            logger=logger,
        )
        
        print("   ✅ F5-TTS готов")
        print(f"      Model: F5-TTS_RUSSIAN")
        print(f"      Device: cuda")
        print(f"      Sample rate: 24000 Hz")
        del f5_tts
        return True
        
    except ImportError:
        print("   ❌ F5-TTS не установлен")
        print("      Установи: pip install f5-tts ruaccent")
        return False
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def check_qwen_availability():
    """Проверка доступности Qwen3-16B-A3B-abliterated-AWQ."""
    print("\n📥 Проверка Qwen3-16B-A3B-abliterated-AWQ...")
    
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "Qwen3-16B-A3B-abliterated-AWQ"
    
    # Проверяем локальную модель
    if model_path.exists():
        config_json = model_path / "config.json"
        if config_json.exists():
            model_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
            if model_files:
                size_gb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (1024**3)
                print(f"   ✅ Модель найдена локально: {model_path}")
                print(f"      Размер: {size_gb:.2f} GB")
                print(f"      Файлов модели: {len(model_files)}")
                return True
    
    # Fallback: проверка в HuggingFace cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_name = "models--warshanks--Qwen3-16B-A3B-abliterated-AWQ"
    model_cache = cache_dir / model_name
    
    if model_cache.exists():
        print(f"   ✅ Модель найдена в кэше: {model_cache}")
        size_gb = sum(f.stat().st_size for f in model_cache.rglob('*') if f.is_file()) / (1024**3)
        print(f"      Размер: {size_gb:.2f} GB")
        return True
    else:
        print(f"   ⚠️  Модель не найдена")
        print(f"      Локальный путь: {model_path}")
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
    results["f5_tts"] = check_f5_tts()
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
    
    critical = ["cuda", "faster_whisper", "f5_tts"]
    if all(results.get(k, False) for k in critical):
        print("✅ Все критичные компоненты готовы к работе!")
        print()
        print("Можешь запускать сервисы:")
        print("  1. vLLM: vllm serve models/Qwen3-16B-A3B-abliterated-AWQ --host 0.0.0.0 --port 8000 --quantization awq --enable-chunked-prefill --enable-prefix-caching")
        print("  2. ASR Gateway: uv run python src/asr_gateway/main.py")
        print("  3. TTS Gateway: uv run python src/tts_gateway/main.py")
        print("  4. Policy Engine: uv run python src/policy_engine/main.py")
        print("  5. FreeSWITCH Bridge: uv run python src/freeswitch_bridge/main.py")
    else:
        print("⚠️  Некоторые компоненты не готовы.")
        print()
        print("Установи недостающие компоненты и повтори проверку.")


if __name__ == "__main__":
    main()

