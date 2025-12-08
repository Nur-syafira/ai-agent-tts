"""
Тестовый скрипт для измерения латентности F5-TTS.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tts_gateway.f5_tts_engine import F5TTSEngine
import logging


def measure_first_audio_latency(tts_engine, text: str, num_runs: int = 5) -> dict:
    """
    Измеряет латентность first-audio для TTS engine.
    
    Args:
        tts_engine: TTS engine (F5TTSEngine)
        text: Текст для синтеза
        num_runs: Количество запусков для усреднения
        
    Returns:
        Словарь с метриками: mean, min, max, std (в мс)
    """
    latencies = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        # Синтезируем аудио
        audio = tts_engine.synthesize(text)
        
        # Время до первого аудио (first-audio latency)
        first_audio_time = time.perf_counter()
        
        latency_ms = (first_audio_time - start_time) * 1000
        latencies.append(latency_ms)
        
        print(f"  Run {i+1}/{num_runs}: {latency_ms:.2f} ms (audio length: {len(audio)} samples)")
    
    return {
        "mean": np.mean(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "std": np.std(latencies),
        "all": latencies,
    }


def test_f5_tts():
    """Тестирует F5-TTS."""
    print("\n" + "="*60)
    print("Testing F5-TTS")
    print("="*60)
    
    try:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Инициализация F5-TTS
        print("Initializing F5-TTS (this may take 10-30 seconds)...")
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
        print("F5-TTS initialized successfully!")
        
        test_phrases = [
            "Добрый день!",
            "Меня зовут администратор медицинского центра.",
            "Как я могу к вам обращаться?",
            "На какую дату вы хотите записаться?",
        ]
        
        results = {}
        for phrase in test_phrases:
            print(f"\nTesting phrase: '{phrase}'")
            metrics = measure_first_audio_latency(f5_tts, phrase)
            results[phrase] = metrics
            print(f"  Mean: {metrics['mean']:.2f} ms")
            print(f"  Min: {metrics['min']:.2f} ms")
            print(f"  Max: {metrics['max']:.2f} ms")
            print(f"  Std: {metrics['std']:.2f} ms")
        
        return results
        
    except Exception as e:
        print(f"Error testing F5-TTS: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Главная функция."""
    print("="*60)
    print("F5-TTS Latency Benchmark")
    print("="*60)
    print("\nThis script measures first-audio latency for F5-TTS.")
    print("\nTarget: F5-TTS first-audio ≤ 150 ms")
    
    # Тестируем F5-TTS
    f5_results = test_f5_tts()
    
    if f5_results:
        # Общая статистика
        f5_all = [m["mean"] for m in f5_results.values()]
        
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"Overall mean: {np.mean(f5_all):.2f} ms")
        print(f"Overall min: {np.min(f5_all):.2f} ms")
        print(f"Overall max: {np.max(f5_all):.2f} ms")
        
        if np.mean(f5_all) <= 150:
            print("\n✅ F5-TTS meets latency requirements (≤150 ms)")
        else:
            print("\n⚠️  F5-TTS exceeds latency requirements (>150 ms)")
    else:
        print("\n❌ Error: Could not test F5-TTS")


if __name__ == "__main__":
    main()
