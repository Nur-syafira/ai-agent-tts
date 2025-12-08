#!/usr/bin/env python3
"""
Стресс-тест VRAM для проверки стабильности работы всех моделей одновременно.

Загружает все модели и мониторит использование VRAM в течение нескольких минут.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any

try:
    import torch
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, OSError):
    NVML_AVAILABLE = False


def get_vram_usage() -> float:
    """Получает текущее использование VRAM в MB."""
    if not NVML_AVAILABLE:
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024 ** 2)
        except:
            pass
        return 0.0
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except:
        return 0.0


def stress_test_vram(duration_minutes: int = 5):
    """
    Стресс-тест VRAM с загрузкой всех моделей.
    
    Args:
        duration_minutes: Длительность теста в минутах
    """
    print("=" * 70)
    print(" " * 20 + "VRAM Stress Test")
    print("=" * 70)
    print(f"Длительность теста: {duration_minutes} минут")
    print()
    
    vram_snapshots: List[Dict[str, Any]] = []
    
    # Начальный VRAM
    vram_initial = get_vram_usage()
    print(f"📊 Начальный VRAM: {vram_initial:.0f} MB")
    
    # Загрузка моделей
    print("\n🔄 Загрузка моделей...")
    
    # 1. ASR
    print("   1. Загрузка ASR (faster-whisper)...")
    from faster_whisper import WhisperModel
    asr_model = WhisperModel("dropbox-dash/faster-whisper-large-v3-turbo", device="cuda", compute_type="int8_float16")
    vram_asr = get_vram_usage()
    vram_snapshots.append({"time": 0, "stage": "asr_loaded", "vram_mb": vram_asr})
    print(f"      VRAM: {vram_asr:.0f} MB (Δ {vram_asr - vram_initial:+.0f} MB)")
    
    # 2. TTS
    print("   2. Загрузка TTS (F5-TTS)...")
    from src.tts_gateway.f5_tts_engine import F5TTSEngine
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    
    project_root = Path(__file__).parent.parent
    f5_tts = F5TTSEngine(
        model_path=str(project_root / "models" / "F5-tts"),
        device="cuda",
        sample_rate=24000,
        use_stress_marks=True,
        logger=logger,
    )
    vram_tts = get_vram_usage()
    vram_snapshots.append({"time": 0, "stage": "tts_loaded", "vram_mb": vram_tts})
    print(f"      VRAM: {vram_tts:.0f} MB (Δ {vram_tts - vram_asr:+.0f} MB)")
    
    # 3. Проверка vLLM (должен быть запущен отдельно)
    print("   3. Проверка vLLM сервера...")
    vram_vllm = get_vram_usage()
    vram_snapshots.append({"time": 0, "stage": "all_loaded", "vram_mb": vram_vllm})
    print(f"      VRAM (все модели): {vram_vllm:.0f} MB")
    
    total_vram_used = vram_vllm - vram_initial
    print(f"\n📊 Суммарное использование VRAM: {total_vram_used / 1024:.2f} GB")
    
    # Мониторинг в течение указанного времени
    print(f"\n⏱️  Мониторинг VRAM в течение {duration_minutes} минут...")
    print("   (Нажмите Ctrl+C для досрочного завершения)\n")
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    interval_seconds = 10  # Замер каждые 10 секунд
    
    try:
        while time.time() < end_time:
            elapsed = time.time() - start_time
            vram_current = get_vram_usage()
            
            vram_snapshots.append({
                "time": elapsed,
                "stage": "monitoring",
                "vram_mb": vram_current,
            })
            
            # Проверка на утечки памяти (рост > 100 MB)
            if len(vram_snapshots) > 1:
                prev_vram = vram_snapshots[-2]["vram_mb"]
                delta = vram_current - prev_vram
                if delta > 100:
                    print(f"   ⚠️  Обнаружен рост VRAM: {prev_vram:.0f} → {vram_current:.0f} MB (Δ {delta:+.0f} MB)")
            
            print(f"   [{elapsed/60:.1f} мин] VRAM: {vram_current:.0f} MB", end="\r")
            time.sleep(interval_seconds)
        
        print()  # Новая строка после завершения
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Тест прерван пользователем")
    
    # Итоговая статистика
    print("\n" + "=" * 70)
    print("📊 Итоговая статистика")
    print("=" * 70)
    
    if vram_snapshots:
        vram_values = [s["vram_mb"] for s in vram_snapshots if s["stage"] == "monitoring"]
        if vram_values:
            vram_min = min(vram_values)
            vram_max = max(vram_values)
            vram_avg = sum(vram_values) / len(vram_values)
            
            print(f"Минимальный VRAM: {vram_min:.0f} MB")
            print(f"Максимальный VRAM: {vram_max:.0f} MB")
            print(f"Средний VRAM: {vram_avg:.0f} MB")
            print(f"Разброс: {vram_max - vram_min:.0f} MB")
            
            if vram_max - vram_min > 500:
                print("\n⚠️  Обнаружены значительные колебания VRAM (>500 MB)")
                print("   Возможна утечка памяти или нестабильность работы")
            else:
                print("\n✅ VRAM стабилен, утечек памяти не обнаружено")
    
    # Выгрузка моделей
    print("\n🔄 Выгрузка моделей...")
    del asr_model
    del f5_tts
    torch.cuda.empty_cache()
    
    vram_final = get_vram_usage()
    print(f"VRAM после выгрузки: {vram_final:.0f} MB")
    
    # Сохранение отчета
    report_dir = Path(__file__).parent.parent / "test_reports"
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"vram_stress_{timestamp}.json"
    
    import json
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "duration_minutes": duration_minutes,
        "vram_initial_mb": vram_initial,
        "vram_final_mb": vram_final,
        "snapshots": vram_snapshots,
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Отчет сохранен: {report_file}")
    print("=" * 70)


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Стресс-тест VRAM для Sales Agent")
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Длительность теста в минутах (по умолчанию 5)",
    )
    
    args = parser.parse_args()
    
    stress_test_vram(duration_minutes=args.duration)


if __name__ == "__main__":
    main()

