#!/usr/bin/env python3
"""
Комплексное пошаговое тестирование Sales Agent системы.

Проверяет все компоненты отдельно с детальными отчетами и мониторингом VRAM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import argparse
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import httpx
import psutil

try:
    import torch
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, OSError):
    NVML_AVAILABLE = False

from src.shared.health import HealthChecker


@dataclass
class TestResult:
    """Результат одного теста."""
    name: str
    status: str  # "success", "error", "warning"
    duration_seconds: float
    vram_before_mb: Optional[float] = None
    vram_after_mb: Optional[float] = None
    vram_peak_mb: Optional[float] = None
    metrics: Dict[str, Any] = None
    error: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.warnings is None:
            self.warnings = []


class SystemTester:
    """Класс для комплексного тестирования системы."""
    
    def __init__(self, stop_on_error: bool = True, continue_on_error: bool = False):
        """
        Инициализация тестера.
        
        Args:
            stop_on_error: Останавливаться при ошибке (по умолчанию True)
            continue_on_error: Продолжать выполнение при ошибках (переопределяет stop_on_error)
        """
        self.results: Dict[str, TestResult] = {}
        self.vram_snapshots: List[Dict[str, Any]] = []
        self.stop_on_error = stop_on_error and not continue_on_error
        self.start_time = time.time()
        
    def get_vram_usage(self) -> Optional[float]:
        """Получает текущее использование VRAM в MB."""
        if not NVML_AVAILABLE:
            try:
                # Fallback через torch
                if torch.cuda.is_available():
                    return torch.cuda.memory_allocated(0) / (1024 ** 2)
            except:
                pass
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024 ** 2)
        except:
            return None
    
    def run_step(self, step_name: str, step_func: Callable) -> bool:
        """
        Выполняет один шаг тестирования с обработкой ошибок.
        
        Args:
            step_name: Имя шага
            step_func: Функция для выполнения
            
        Returns:
            True если шаг выполнен успешно, False если ошибка
        """
        print(f"\n{'='*70}")
        print(f"🔍 {step_name}")
        print(f"{'='*70}")
        
        vram_before = self.get_vram_usage()
        start_time = time.time()
        result = TestResult(name=step_name, status="success", duration_seconds=0.0)
        
        try:
            # Выполняем шаг
            if asyncio.iscoroutinefunction(step_func):
                step_result = asyncio.run(step_func())
            else:
                step_result = step_func()
            
            duration = time.time() - start_time
            vram_after = self.get_vram_usage()
            
            result.duration_seconds = duration
            result.vram_before_mb = vram_before
            result.vram_after_mb = vram_after
            
            if isinstance(step_result, dict):
                result.metrics.update(step_result)
            
            # Сохраняем snapshot VRAM
            self.vram_snapshots.append({
                "step": step_name,
                "timestamp": datetime.now().isoformat(),
                "vram_mb": vram_after,
                "vram_delta_mb": (vram_after - vram_before) if vram_before and vram_after else None,
            })
            
            print(f"✅ {step_name} - успешно ({duration:.2f}s)")
            if vram_before and vram_after:
                delta = vram_after - vram_before
                print(f"   VRAM: {vram_before:.0f} MB → {vram_after:.0f} MB (Δ {delta:+.0f} MB)")
            
            result.status = "success"
            self.results[step_name] = result
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            vram_after = self.get_vram_usage()
            
            result.duration_seconds = duration
            result.status = "error"
            result.error = str(e)
            result.vram_before_mb = vram_before
            result.vram_after_mb = vram_after
            
            print(f"❌ {step_name} - ошибка ({duration:.2f}s)")
            print(f"   Ошибка: {e}")
            print(f"   Traceback:")
            traceback.print_exc()
            
            self.results[step_name] = result
            
            if self.stop_on_error:
                print(f"\n⚠️  Остановка тестирования из-за ошибки в шаге '{step_name}'")
                print(f"   Используйте --continue-on-error чтобы продолжить при ошибках")
                return False
            
            return False
    
    def test_environment(self) -> Dict[str, Any]:
        """Проверка окружения и зависимостей."""
        metrics = {}
        
        # Python версия
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        metrics["python_version"] = python_version
        if sys.version_info < (3, 12):
            raise RuntimeError(f"Требуется Python 3.12+, текущая версия: {python_version}")
        print(f"   Python: {python_version} ✅")
        
        # CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна")
        
        cuda_version = torch.version.cuda
        metrics["cuda_version"] = cuda_version
        print(f"   CUDA: {cuda_version} ✅")
        
        # GPU информация
        gpu_count = torch.cuda.device_count()
        metrics["gpu_count"] = gpu_count
        print(f"   GPU устройств: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_vram_gb = props.total_memory / (1024 ** 3)
            metrics[f"gpu_{i}_name"] = props.name
            metrics[f"gpu_{i}_vram_gb"] = total_vram_gb
            print(f"   GPU {i}: {props.name}")
            print(f"      VRAM: {total_vram_gb:.2f} GB")
        
        # Проверка пакетов
        packages = {
            "torch": torch.__version__,
            "faster-whisper": None,
            "f5-tts": None,
            "vllm": None,
            "huggingface-hub": None,
        }
        
        try:
            import faster_whisper
            packages["faster-whisper"] = getattr(faster_whisper, "__version__", "unknown")
        except ImportError:
            raise RuntimeError("faster-whisper не установлен")
        
        try:
            import f5_tts
            packages["f5-tts"] = getattr(f5_tts, "__version__", "unknown")
        except ImportError:
            raise RuntimeError("f5-tts не установлен")
        
        try:
            import vllm
            packages["vllm"] = getattr(vllm, "__version__", "unknown")
        except ImportError:
            # vllm может быть не установлен, если запускается отдельно
            packages["vllm"] = None
            print("   ⚠️  vllm не установлен (это нормально если запускается отдельно)")
        
        try:
            import huggingface_hub
            packages["huggingface-hub"] = getattr(huggingface_hub, "__version__", "unknown")
        except ImportError:
            pass  # Не критично
        
        metrics["packages"] = packages
        print(f"   Пакеты: {', '.join(f'{k}={v}' for k, v in packages.items() if v)} ✅")
        
        return metrics
    
    def test_local_models(self) -> Dict[str, Any]:
        """Проверка локальных моделей на диске."""
        metrics = {}
        project_root = Path(__file__).parent.parent
        
        # Проверка F5-TTS
        f5_path = project_root / "models" / "F5-tts"
        if not f5_path.exists():
            raise RuntimeError(f"F5-TTS модель не найдена: {f5_path}")
        
        f5_files = list(f5_path.glob("*.pt")) + list(f5_path.glob("*.safetensors"))
        if not f5_files:
            raise RuntimeError(f"Файлы модели F5-TTS не найдены в {f5_path}")
        
        f5_size_mb = sum(f.stat().st_size for f in f5_files) / (1024 ** 2)
        metrics["f5_tts_path"] = str(f5_path)
        metrics["f5_tts_size_mb"] = f5_size_mb
        metrics["f5_tts_files"] = [f.name for f in f5_files]
        print(f"   F5-TTS: {f5_path}")
        print(f"      Размер: {f5_size_mb:.2f} MB")
        print(f"      Файлы: {', '.join(f.name for f in f5_files)} ✅")
        
        # Проверка Qwen3
        qwen_path = project_root / "models" / "Qwen3-16B-A3B-abliterated-AWQ"
        if not qwen_path.exists():
            raise RuntimeError(f"Qwen3 модель не найдена: {qwen_path}")
        
        config_json = qwen_path / "config.json"
        if not config_json.exists():
            raise RuntimeError(f"config.json не найден в {qwen_path}")
        
        # Подсчет размера модели
        model_files = list(qwen_path.glob("*.safetensors")) + list(qwen_path.glob("*.bin"))
        if not model_files:
            raise RuntimeError(f"Файлы модели Qwen3 не найдены в {qwen_path}")
        
        qwen_size_gb = sum(f.stat().st_size for f in qwen_path.rglob("*") if f.is_file()) / (1024 ** 3)
        metrics["qwen_path"] = str(qwen_path)
        metrics["qwen_size_gb"] = qwen_size_gb
        metrics["qwen_model_files_count"] = len(model_files)
        print(f"   Qwen3: {qwen_path}")
        print(f"      Размер: {qwen_size_gb:.2f} GB")
        print(f"      Файлов модели: {len(model_files)} ✅")
        
        return metrics
    
    def run_all_tests(self, steps: Optional[List[str]] = None):
        """
        Выполняет все тесты последовательно.
        
        Args:
            steps: Список шагов для выполнения (если None, выполняются все)
        """
        all_steps = [
            ("environment", self.test_environment),
            ("local_models", self.test_local_models),
            ("asr_model", self.test_asr_model),
            ("tts_model", self.test_tts_model),
            ("llm_model", self.test_llm_model),
            ("vram_all_models", self.test_vram_all_models),
            ("redis", self.test_redis),
            ("vllm_server", self.test_vllm_server),
            ("asr_gateway", self.test_asr_gateway),
            ("tts_gateway", self.test_tts_gateway),
            ("policy_engine", self.test_policy_engine),
            ("freeswitch_bridge", self.test_freeswitch_bridge),
            ("e2e_dialog", self.test_e2e_dialog),
        ]
        
        if steps:
            # Фильтруем только запрошенные шаги
            step_dict = {name: func for name, func in all_steps}
            all_steps = [(name, step_dict[name]) for name in steps if name in step_dict]
        
        print("=" * 70)
        print(" " * 20 + "Sales Agent - System Test")
        print("=" * 70)
        print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Шагов для выполнения: {len(all_steps)}")
        print()
        
        for step_name, step_func in all_steps:
            success = self.run_step(step_name, step_func)
            if not success and self.stop_on_error:
                break
        
        # Генерация отчета
        self.generate_report()
    
    def test_asr_model(self) -> Dict[str, Any]:
        """Загрузка и тест ASR модели (faster-whisper) с измерением VRAM и латентности."""
        import numpy as np
        from faster_whisper import WhisperModel
        
        metrics = {}
        
        print("   Загрузка faster-whisper large-v3-turbo (Dropbox)...")
        model = WhisperModel(
            "dropbox-dash/faster-whisper-large-v3-turbo",
            device="cuda",
            compute_type="int8_float16",
        )
        
        vram_after_load = self.get_vram_usage()
        metrics["vram_after_load_mb"] = vram_after_load
        
        # Тестовое распознавание
        print("   Тестовое распознавание...")
        test_audio = np.random.randn(16000).astype(np.float32)  # 1 секунда @ 16kHz
        
        start_time = time.time()
        segments, info = model.transcribe(test_audio, beam_size=1)
        # Получаем первый сегмент
        first_segment = next(segments, None)
        latency_ms = (time.time() - start_time) * 1000
        
        metrics["latency_ms"] = latency_ms
        metrics["language"] = info.language if hasattr(info, 'language') else None
        
        print(f"   Латентность распознавания: {latency_ms:.2f} мс")
        
        # Выгрузка модели
        del model
        torch.cuda.empty_cache()
        
        vram_after_unload = self.get_vram_usage()
        metrics["vram_after_unload_mb"] = vram_after_unload
        
        return metrics
    
    def test_tts_model(self) -> Dict[str, Any]:
        """Загрузка и тест TTS модели (F5-TTS) с измерением VRAM и латентности first-audio."""
        from src.tts_gateway.f5_tts_engine import F5TTSEngine
        import logging
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)
        
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "F5-tts"
        
        metrics = {}
        
        print(f"   Загрузка F5-TTS из {model_path}...")
        f5_tts = F5TTSEngine(
            model_path=str(model_path),
            device="cuda",
            sample_rate=24000,
            use_stress_marks=True,
            logger=logger,
        )
        
        vram_after_load = self.get_vram_usage()
        metrics["vram_after_load_mb"] = vram_after_load
        
        # Тестовый синтез
        test_text = "Добрый день! Как я могу вам помочь?"
        print(f"   Тестовый синтез: '{test_text}'...")
        
        start_time = time.time()
        audio = f5_tts.synthesize(test_text)
        latency_ms = (time.time() - start_time) * 1000
        
        metrics["latency_ms"] = latency_ms
        metrics["audio_length_samples"] = len(audio)
        metrics["audio_duration_sec"] = len(audio) / 24000
        
        print(f"   Латентность синтеза: {latency_ms:.2f} мс")
        print(f"   Длина аудио: {len(audio)} сэмплов ({metrics['audio_duration_sec']:.2f} сек)")
        
        # Выгрузка модели
        del f5_tts
        torch.cuda.empty_cache()
        
        vram_after_unload = self.get_vram_usage()
        metrics["vram_after_unload_mb"] = vram_after_unload
        
        return metrics
    
    def test_llm_model(self) -> Dict[str, Any]:
        """Проверка локальной модели Qwen3 и конфигурации."""
        import json
        
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "Qwen3-16B-A3B-abliterated-AWQ"
        
        metrics = {}
        
        # Проверка config.json
        config_json = model_path / "config.json"
        if not config_json.exists():
            raise RuntimeError(f"config.json не найден в {model_path}")
        
        with open(config_json, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        metrics["model_type"] = config.get("model_type")
        metrics["hidden_size"] = config.get("hidden_size")
        metrics["num_hidden_layers"] = config.get("num_hidden_layers")
        metrics["num_experts"] = config.get("num_experts")
        metrics["num_experts_per_tok"] = config.get("num_experts_per_tok")
        
        print(f"   Тип модели: {metrics['model_type']}")
        print(f"   Архитектура: MoE с {metrics['num_experts']} экспертами")
        print(f"   Экспертов на токен: {metrics['num_experts_per_tok']}")
        
        # Проверка файлов модели
        model_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
        metrics["model_files_count"] = len(model_files)
        
        if not model_files:
            raise RuntimeError(f"Файлы модели не найдены в {model_path}")
        
        print(f"   Файлов модели: {len(model_files)}")
        
        # Проверка что vLLM может найти модель (не загружаем, только проверяем путь)
        metrics["model_path"] = str(model_path)
        print(f"   Путь модели: {model_path} ✅")
        
        return metrics
    
    async def test_vram_all_models(self) -> Dict[str, Any]:
        """Одновременная загрузка всех моделей с детальным мониторингом VRAM."""
        import subprocess
        import signal
        
        metrics = {}
        vram_snapshots = []
        
        vram_initial = self.get_vram_usage()
        vram_snapshots.append({"stage": "initial", "vram_mb": vram_initial})
        metrics["vram_initial_mb"] = vram_initial
        
        print(f"   Начальный VRAM: {vram_initial:.0f} MB")
        
        # 1. Загрузка ASR
        print("\n   1. Загрузка ASR (faster-whisper)...")
        from faster_whisper import WhisperModel
        asr_model = WhisperModel("dropbox-dash/faster-whisper-large-v3-turbo", device="cuda", compute_type="int8_float16")
        vram_asr = self.get_vram_usage()
        vram_snapshots.append({"stage": "asr_loaded", "vram_mb": vram_asr})
        metrics["vram_asr_mb"] = vram_asr
        print(f"      VRAM после ASR: {vram_asr:.0f} MB (Δ {vram_asr - vram_initial:+.0f} MB)")
        
        # 2. Загрузка TTS
        print("\n   2. Загрузка TTS (F5-TTS)...")
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
        vram_tts = self.get_vram_usage()
        vram_snapshots.append({"stage": "tts_loaded", "vram_mb": vram_tts})
        metrics["vram_tts_mb"] = vram_tts
        print(f"      VRAM после TTS: {vram_tts:.0f} MB (Δ {vram_tts - vram_asr:+.0f} MB)")
        
        # 3. Запуск vLLM сервера (в фоне)
        print("\n   3. Запуск vLLM сервера...")
        print("      (Проверяем что сервер уже запущен или запускаем в фоне)")
        
        # Проверяем запущен ли сервер
        vllm_running = False
        try:
            async def check_vllm():
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get("http://localhost:8000/v1/models")
                    return response.status_code == 200
            vllm_running = asyncio.run(check_vllm())
        except:
            pass
        
        if not vllm_running:
            print("      ⚠️  vLLM сервер не запущен. Запустите его отдельно:")
            print("         vllm serve models/Qwen3-16B-A3B-abliterated-AWQ --host 0.0.0.0 --port 8000 --quantization awq")
            metrics["vllm_warning"] = "vLLM сервер не запущен"
        else:
            print("      ✅ vLLM сервер запущен")
            # Даем время на загрузку модели если только что запустился
            await asyncio.sleep(5)
        
        vram_vllm = self.get_vram_usage()
        vram_snapshots.append({"stage": "vllm_loaded", "vram_mb": vram_vllm})
        metrics["vram_vllm_mb"] = vram_vllm
        print(f"      VRAM после vLLM: {vram_vllm:.0f} MB (Δ {vram_vllm - vram_tts:+.0f} MB)")
        
        # Итоговая статистика
        total_vram_used = vram_vllm - vram_initial
        metrics["vram_total_used_mb"] = total_vram_used
        
        # Получаем общую VRAM GPU
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            metrics["vram_total_gb"] = total_vram_gb
            free_vram_gb = (torch.cuda.get_device_properties(0).total_memory - vram_vllm * 1024 ** 2) / (1024 ** 3)
            metrics["vram_free_gb"] = free_vram_gb
            
            print(f"\n   📊 Итоговая статистика:")
            print(f"      Всего VRAM: {total_vram_gb:.2f} GB")
            print(f"      Использовано: {total_vram_used / 1024:.2f} GB")
            print(f"      Свободно: {free_vram_gb:.2f} GB")
            
            if free_vram_gb < 1.0:
                metrics["warning"] = f"Мало свободной VRAM: {free_vram_gb:.2f} GB (рекомендуется минимум 1 GB)"
                print(f"      ⚠️  {metrics['warning']}")
        
        # Выгрузка моделей
        print("\n   Выгрузка моделей...")
        del asr_model
        del f5_tts
        torch.cuda.empty_cache()
        
        vram_final = self.get_vram_usage()
        vram_snapshots.append({"stage": "unloaded", "vram_mb": vram_final})
        metrics["vram_final_mb"] = vram_final
        metrics["vram_snapshots"] = vram_snapshots
        
        return metrics
    
    async def test_redis(self) -> Dict[str, Any]:
        """Проверка подключения и работы Redis."""
        import redis.asyncio as aioredis
        import os
        
        metrics = {}
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        metrics["redis_url"] = redis_url
        
        print(f"   Подключение к Redis: {redis_url}...")
        
        try:
            client = await aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            
            # Ping
            pong = await client.ping()
            if not pong:
                raise RuntimeError("Redis ping failed")
            metrics["ping_success"] = True
            print("   ✅ Ping успешен")
            
            # Тест записи/чтения
            test_key = "test_system_check"
            test_value = "test_value_123"
            await client.set(test_key, test_value, ex=10)
            read_value = await client.get(test_key)
            
            if read_value != test_value:
                raise RuntimeError(f"Redis read/write test failed: expected {test_value}, got {read_value}")
            
            await client.delete(test_key)
            metrics["read_write_success"] = True
            print("   ✅ Запись/чтение успешны")
            
            # Проверка информации о Redis
            info = await client.info("memory")
            metrics["redis_memory_used_mb"] = int(info.get("used_memory", 0)) / (1024 ** 2)
            print(f"   Использование памяти Redis: {metrics['redis_memory_used_mb']:.2f} MB")
            
            await client.close()
            
        except Exception as e:
            raise RuntimeError(f"Redis connection failed: {e}") from e
        
        return metrics
    
    async def test_vllm_server(self) -> Dict[str, Any]:
        """Проверка vLLM сервера с измерением TTFT и latency."""
        from openai import AsyncOpenAI
        
        metrics = {}
        
        base_url = "http://localhost:8000/v1"
        print(f"   Проверка vLLM сервера: {base_url}...")
        
        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        
        # Проверка что сервер запущен
        try:
            models = await client.models.list()
            if not models.data:
                raise RuntimeError("vLLM сервер не вернул модели")
            
            model_name = models.data[0].id
            metrics["model_name"] = model_name
            print(f"   ✅ Сервер запущен, модель: {model_name}")
            
        except Exception as e:
            raise RuntimeError(f"vLLM сервер недоступен: {e}") from e
        
        # Тестовая генерация
        test_messages = [
            {"role": "system", "content": "Ты помощник."},
            {"role": "user", "content": "Привет! Скажи коротко: как дела?"}
        ]
        
        print("   Тестовая генерация...")
        start_time = time.time()
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=test_messages,
            max_tokens=50,
            temperature=0.7,
        )
        
        total_time = time.time() - start_time
        ttft_ms = response.response_headers.get("x-first-token-ms") if hasattr(response, 'response_headers') else None
        
        metrics["total_latency_ms"] = total_time * 1000
        metrics["ttft_ms"] = float(ttft_ms) if ttft_ms else None
        metrics["response_text"] = response.choices[0].message.content[:50]
        metrics["tokens_generated"] = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else None
        
        print(f"   ✅ Генерация успешна")
        print(f"      Общая латентность: {metrics['total_latency_ms']:.2f} мс")
        if metrics["ttft_ms"]:
            print(f"      TTFT: {metrics['ttft_ms']:.2f} мс")
        
        # Проверка VRAM vLLM
        vram_vllm = self.get_vram_usage()
        metrics["vram_usage_mb"] = vram_vllm
        print(f"      VRAM: {vram_vllm:.0f} MB")
        
        return metrics
    
    async def test_asr_gateway(self) -> Dict[str, Any]:
        """Проверка ASR Gateway health и readiness."""
        metrics = {}
        
        base_url = "http://localhost:8001"
        print(f"   Проверка ASR Gateway: {base_url}...")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Health check
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code != 200:
                    raise RuntimeError(f"Health check failed: HTTP {response.status_code}")
                metrics["health_status"] = "ok"
                print("   ✅ Health check успешен")
            except Exception as e:
                raise RuntimeError(f"ASR Gateway недоступен: {e}") from e
            
            # Readiness check
            try:
                response = await client.get(f"{base_url}/ready")
                metrics["ready"] = response.status_code == 200
                print(f"   ✅ Readiness: {'ready' if metrics['ready'] else 'not ready'}")
            except Exception as e:
                metrics["ready"] = False
                metrics["ready_error"] = str(e)
                print(f"   ⚠️  Readiness check failed: {e}")
        
        return metrics
    
    async def test_tts_gateway(self) -> Dict[str, Any]:
        """Проверка TTS Gateway с тестовым синтезом."""
        metrics = {}
        
        base_url = "http://localhost:8002"
        print(f"   Проверка TTS Gateway: {base_url}...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Health check
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code != 200:
                    raise RuntimeError(f"Health check failed: HTTP {response.status_code}")
                metrics["health_status"] = "ok"
                print("   ✅ Health check успешен")
            except Exception as e:
                raise RuntimeError(f"TTS Gateway недоступен: {e}") from e
            
            # Тестовый синтез
            test_text = "Добрый день!"
            print(f"   Тестовый синтез: '{test_text}'...")
            
            start_time = time.time()
            response = await client.post(
                f"{base_url}/synthesize",
                json={"text": test_text},
            )
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                raise RuntimeError(f"Synthesis failed: HTTP {response.status_code}")
            
            audio_data = response.content
            sample_rate = int(response.headers.get("X-Sample-Rate", "24000"))
            channels = int(response.headers.get("X-Channels", "1"))
            
            metrics["latency_ms"] = latency_ms
            metrics["audio_size_bytes"] = len(audio_data)
            metrics["sample_rate"] = sample_rate
            metrics["channels"] = channels
            
            print(f"   ✅ Синтез успешен")
            print(f"      Латентность: {latency_ms:.2f} мс")
            print(f"      Размер аудио: {len(audio_data)} байт")
            print(f"      Sample rate: {sample_rate} Hz")
        
        return metrics
    
    async def test_policy_engine(self) -> Dict[str, Any]:
        """Проверка Policy Engine с тестовым диалогом."""
        metrics = {}
        
        base_url = "http://localhost:8003"
        print(f"   Проверка Policy Engine: {base_url}...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Health check
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code != 200:
                    raise RuntimeError(f"Health check failed: HTTP {response.status_code}")
                metrics["health_status"] = "ok"
                print("   ✅ Health check успешен")
            except Exception as e:
                raise RuntimeError(f"Policy Engine недоступен: {e}") from e
            
            # Тестовый диалог
            test_session_id = f"test-{int(time.time())}"
            test_message = "Добрый день, хочу записаться на МРТ"
            
            print(f"   Тестовый диалог запрос...")
            start_time = time.time()
            
            response = await client.post(
                f"{base_url}/dialog",
                json={
                    "session_id": test_session_id,
                    "user_message": test_message,
                },
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                raise RuntimeError(f"Dialog request failed: HTTP {response.status_code}")
            
            data = response.json()
            metrics["latency_ms"] = latency_ms
            metrics["agent_message"] = data.get("agent_message", "")[:50]
            metrics["current_state"] = data.get("current_state")
            metrics["is_complete"] = data.get("is_complete", False)
            metrics["slots_count"] = len(data.get("slots", {}))
            
            print(f"   ✅ Диалог успешен")
            print(f"      Латентность: {latency_ms:.2f} мс")
            print(f"      Состояние: {metrics['current_state']}")
            print(f"      Слотов заполнено: {metrics['slots_count']}")
        
        return metrics
    
    async def test_freeswitch_bridge(self) -> Dict[str, Any]:
        """Проверка FreeSWITCH Bridge через API и WebSocket."""
        metrics = {}
        
        base_url = "http://localhost:8004"
        print(f"   Проверка FreeSWITCH Bridge: {base_url}...")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Health check
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code != 200:
                    raise RuntimeError(f"Health check failed: HTTP {response.status_code}")
                metrics["health_status"] = "ok"
                print("   ✅ Health check успешен")
            except Exception as e:
                raise RuntimeError(f"FreeSWITCH Bridge недоступен: {e}") from e
            
            # Проверка конфигурации API ключа
            from src.shared.config_loader import load_and_validate_config
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent / "src" / "freeswitch_bridge" / "config.yaml"
            # Проверяем что конфиг существует (не загружаем полностью, чтобы не зависеть от структуры)
            if config_path.exists():
                metrics["config_exists"] = True
                print("   ✅ Конфигурация найдена")
            else:
                metrics["config_exists"] = False
                print("   ⚠️  Конфигурация не найдена")
        
        # WebSocket тест (симуляция подключения)
        try:
            import websockets
            ws_url = f"ws://localhost:8004/ws"
            print(f"   Проверка WebSocket: {ws_url}...")
            
            async def test_ws():
                try:
                    async with websockets.connect(ws_url, timeout=2.0) as ws:
                        await ws.ping()
                        return True
                except:
                    return False
            
            ws_available = await test_ws()
            metrics["websocket_available"] = ws_available
            if ws_available:
                print("   ✅ WebSocket доступен")
            else:
                print("   ⚠️  WebSocket недоступен (это нормально если нет активных звонков)")
        except ImportError:
            metrics["websocket_test"] = "websockets not installed"
            print("   ⚠️  websockets не установлен, пропускаем WebSocket тест")
        except Exception as e:
            metrics["websocket_error"] = str(e)
            print(f"   ⚠️  WebSocket тест failed: {e}")
        
        return metrics
    
    async def test_e2e_dialog(self) -> Dict[str, Any]:
        """Полный E2E тест диалога с детальными метриками каждого шага."""
        from scripts.simulate_dialog import DialogSimulator
        
        metrics = {}
        
        print("   Запуск E2E теста диалога...")
        
        simulator = DialogSimulator(
            policy_url="http://localhost:8003",
            session_id=f"e2e-test-{int(time.time())}",
        )
        
        # Проверка сервисов перед запуском
        if not await simulator.check_services():
            raise RuntimeError("Policy Engine недоступен для E2E теста")
        
        # Запускаем короткий диалог (5-10 ходов)
        start_time = time.time()
        
        try:
            # Первый ход
            agent_msg, state, is_complete, slots = await simulator.send_message("")
            metrics["first_turn_latency_ms"] = simulator.metrics.avg_response_time_ms
            
            # Несколько ходов для теста
            test_messages = [
                "Меня зовут Иван Петров",
                "У меня болит голова",
                "Хочу записаться на завтра в 15:00",
            ]
            
            turn_latencies = []
            for msg in test_messages:
                turn_start = time.time()
                agent_msg, state, is_complete, slots = await simulator.send_message(msg)
                turn_latency = (time.time() - turn_start) * 1000
                turn_latencies.append(turn_latency)
            
            total_time = time.time() - start_time
            
            metrics["total_turns"] = simulator.metrics.total_turns
            metrics["total_time_seconds"] = total_time
            metrics["avg_turn_latency_ms"] = sum(turn_latencies) / len(turn_latencies) if turn_latencies else 0
            metrics["turn_latencies_ms"] = turn_latencies
            metrics["fsm_transitions"] = simulator.metrics.fsm_transitions
            metrics["slots_filled"] = simulator.metrics.slots_filled
            metrics["is_complete"] = is_complete
            
            print(f"   ✅ E2E тест завершен")
            print(f"      Ходов: {metrics['total_turns']}")
            print(f"      Общее время: {total_time:.2f} сек")
            print(f"      Средняя латентность хода: {metrics['avg_turn_latency_ms']:.2f} мс")
            print(f"      Слотов заполнено: {metrics['slots_filled']}")
            
        except Exception as e:
            raise RuntimeError(f"E2E тест failed: {e}") from e
        
        return metrics
    
    def generate_report(self):
        """Генерация детального отчета."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(__file__).parent.parent / "test_reports"
        report_dir.mkdir(exist_ok=True)
        
        md_path = report_dir / f"test_report_{timestamp}.md"
        json_path = report_dir / f"test_report_{timestamp}.json"
        
        # Генерируем JSON отчет
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - self.start_time,
            "results": {name: asdict(result) for name, result in self.results.items()},
            "vram_snapshots": self.vram_snapshots,
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Генерируем Markdown отчет
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Sales Agent - Test Report\n\n")
            f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Длительность:** {time.time() - self.start_time:.2f} секунд\n\n")
            
            # Сводка
            total = len(self.results)
            success = sum(1 for r in self.results.values() if r.status == "success")
            errors = sum(1 for r in self.results.values() if r.status == "error")
            
            f.write(f"## Сводка\n\n")
            f.write(f"- Всего проверок: {total}\n")
            f.write(f"- Успешно: {success} ✅\n")
            f.write(f"- Ошибок: {errors} ❌\n\n")
            
            # Таблица использования VRAM
            if self.vram_snapshots:
                f.write(f"## Использование VRAM\n\n")
                f.write(f"| Этап | VRAM (MB) | Δ (MB) |\n")
                f.write(f"|------|-----------|--------|\n")
                prev_vram = None
                for snapshot in self.vram_snapshots:
                    vram = snapshot.get("vram_mb")
                    delta = snapshot.get("vram_delta_mb")
                    if vram:
                        delta_str = f"{delta:+.0f}" if delta else "-"
                        f.write(f"| {snapshot.get('step', 'unknown')} | {vram:.0f} | {delta_str} |\n")
                        prev_vram = vram
                f.write(f"\n")
            
            # Детали по шагам
            f.write(f"## Детали проверок\n\n")
            for name, result in self.results.items():
                status_emoji = "✅" if result.status == "success" else "❌"
                f.write(f"### {status_emoji} {name}\n\n")
                f.write(f"- **Статус:** {result.status}\n")
                f.write(f"- **Время:** {result.duration_seconds:.2f}s\n")
                
                if result.vram_before_mb and result.vram_after_mb:
                    delta = result.vram_after_mb - result.vram_before_mb
                    f.write(f"- **VRAM:** {result.vram_before_mb:.0f} MB → {result.vram_after_mb:.0f} MB (Δ {delta:+.0f} MB)\n")
                    if result.vram_peak_mb:
                        f.write(f"- **VRAM пик:** {result.vram_peak_mb:.0f} MB\n")
                
                if result.metrics:
                    f.write(f"- **Метрики:**\n")
                    for k, v in result.metrics.items():
                        if isinstance(v, (int, float)):
                            f.write(f"  - `{k}`: {v}\n")
                        elif isinstance(v, str):
                            f.write(f"  - `{k}`: {v}\n")
                        elif isinstance(v, list):
                            f.write(f"  - `{k}`: {len(v)} элементов\n")
                        else:
                            f.write(f"  - `{k}`: {v}\n")
                
                if result.warnings:
                    f.write(f"- **Предупреждения:**\n")
                    for warning in result.warnings:
                        f.write(f"  - ⚠️ {warning}\n")
                
                if result.error:
                    f.write(f"- **Ошибка:** `{result.error}`\n")
                
                f.write(f"\n")
            
            # Рекомендации
            if errors > 0:
                f.write(f"## Рекомендации\n\n")
                f.write(f"Обнаружены ошибки в следующих проверках:\n\n")
                for name, result in self.results.items():
                    if result.status == "error":
                        f.write(f"- **{name}**: {result.error}\n")
                f.write(f"\n")
            
            # Следующие шаги
            f.write(f"## Следующие шаги\n\n")
            if errors == 0:
                f.write(f"✅ Все проверки пройдены успешно! Система готова к работе.\n\n")
            else:
                f.write(f"⚠️ Исправьте обнаруженные ошибки перед использованием системы.\n\n")
            
            f.write(f"Для повторного запуска тестов:\n")
            f.write(f"```bash\n")
            f.write(f"uv run python scripts/test_system.py\n")
            f.write(f"```\n")
        
        print(f"\n{'='*70}")
        print(f"📊 Отчеты сохранены:")
        print(f"   Markdown: {md_path}")
        print(f"   JSON: {json_path}")
        print(f"{'='*70}")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Комплексное тестирование Sales Agent")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Продолжать выполнение при ошибках",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Список шагов для выполнения через запятую (например: environment,models,vram)",
    )
    
    args = parser.parse_args()
    
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
    
    tester = SystemTester(continue_on_error=args.continue_on_error)
    tester.run_all_tests(steps=steps)


if __name__ == "__main__":
    main()

