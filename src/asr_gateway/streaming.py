"""
Streaming ASR с использованием RealtimeSTT и Silero VAD.
"""

import torch
import numpy as np
from typing import Optional, Callable, Dict, Any
from RealtimeSTT import AudioToTextRecorder
import logging
from pathlib import Path


class StreamingASR:
    """Класс для потокового распознавания речи."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "ru",
        beam_size: int = 1,
        vad_enabled: bool = True,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: int = 30,
        min_silence_duration_ms: int = 500,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация Streaming ASR.
        
        Args:
            model_name: Название faster-whisper модели
            device: Устройство ('cuda' или 'cpu')
            compute_type: Тип вычислений (int8_float16, float16, float32)
            language: Язык распознавания
            beam_size: Размер beam для декодирования (1 = fastest)
            vad_enabled: Включить VAD
            vad_threshold: Порог VAD (0.0-1.0)
            min_speech_duration_ms: Минимальная длительность речи
            max_speech_duration_s: Максимальная длительность речи
            min_silence_duration_ms: Минимальная тишина для endpointing
            logger: Logger объект
            
        Raises:
            RuntimeError: Если CUDA недоступна при device='cuda'
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Guard-проверка CUDA
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required but not available. "
                "Please check your GPU drivers and PyTorch installation."
            )
        
        self.device = device
        self.model_name = model_name
        self.compute_type = compute_type
        self.language = language
        
        self.logger.info(
            "Initializing StreamingASR",
            extra={
                "context": {
                    "model": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "language": language,
                }
            },
        )
        
        # Инициализация RealtimeSTT recorder
        self.recorder = AudioToTextRecorder(
            model=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            silero_use_onnx=True,  # Быстрее на GPU
            silero_sensitivity=vad_threshold,
            min_length_of_recording=min_speech_duration_ms / 1000.0,
            min_gap_between_recordings=min_silence_duration_ms / 1000.0,
            enable_realtime_transcription=True,
            realtime_processing_pause=0.05,  # 50ms паузы между обработками
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
        )
        
        self.logger.info("StreamingASR initialized successfully")
        
    def _on_recording_start(self):
        """Callback при начале записи (VAD обнаружил речь)."""
        self.logger.debug("Recording started (VAD detected speech)")
        
    def _on_recording_stop(self):
        """Callback при остановке записи (VAD обнаружил тишину)."""
        self.logger.debug("Recording stopped (VAD detected silence)")

    def transcribe_stream(
        self,
        on_partial_transcript: Optional[Callable[[str], None]] = None,
        on_final_transcript: Optional[Callable[[str], None]] = None,
    ):
        """
        Запускает потоковое распознавание.
        
        Args:
            on_partial_transcript: Callback для частичных транскриптов
            on_final_transcript: Callback для финальных транскриптов
        """
        self.logger.info("Starting streaming transcription")
        
        try:
            while True:
                # Получаем частичный транскрипт
                if on_partial_transcript:
                    partial = self.recorder.text()
                    if partial:
                        on_partial_transcript(partial)
                
                # Получаем финальный транскрипт (после endpointing)
                full_text = self.recorder.text()
                if full_text and on_final_transcript:
                    on_final_transcript(full_text)
                    
        except KeyboardInterrupt:
            self.logger.info("Streaming transcription stopped by user")
        except Exception as e:
            self.logger.error(f"Error in streaming transcription: {e}", exc_info=True)
            raise

    def transcribe_audio_chunk(
        self, audio_chunk: np.ndarray, sample_rate: int = 16000
    ) -> Optional[str]:
        """
        Транскрибирует один аудио чанк.
        
        Args:
            audio_chunk: Numpy array с аудио данными
            sample_rate: Частота дискретизации
            
        Returns:
            Транскрипт или None
        """
        try:
            # Подаём аудио в recorder
            self.recorder.feed_audio(audio_chunk)
            
            # Получаем текущий транскрипт
            text = self.recorder.text()
            return text if text else None
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio chunk: {e}", exc_info=True)
            return None

    def warmup(self):
        """
        Прогревает модель тестовым аудио для снижения латентности первого запроса.
        """
        self.logger.info("Warming up ASR model...")
        
        try:
            # Генерируем тестовое аудио (1 секунда тишины)
            dummy_audio = np.zeros(16000, dtype=np.float32)
            self.transcribe_audio_chunk(dummy_audio)
            
            self.logger.info("ASR model warmed up successfully")
            
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}")

    def shutdown(self):
        """Корректное завершение работы recorder."""
        self.logger.info("Shutting down StreamingASR")
        try:
            self.recorder.shutdown()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

