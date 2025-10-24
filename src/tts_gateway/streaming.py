"""
Streaming TTS с Kokoro-82M и Piper fallback.
"""

import numpy as np
from typing import Optional, Generator
import logging
from pathlib import Path
import subprocess
import os


class PiperTTS:
    """Fallback TTS с использованием Piper."""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        sample_rate: int = 22050,
        speed: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация Piper TTS.
        
        Args:
            model_path: Путь к модели .onnx
            config_path: Путь к config.json
            sample_rate: Частота дискретизации
            speed: Скорость речи
            logger: Logger
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.sample_rate = sample_rate
        self.speed = speed
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")

    def synthesize(self, text: str) -> np.ndarray:
        """
        Синтезирует речь из текста.
        
        Args:
            text: Текст для синтеза
            
        Returns:
            Numpy array с аудио (16-bit PCM, mono)
        """
        try:
            # Запускаем Piper через subprocess
            cmd = ["piper", "--model", str(self.model_path), "--output-raw"]
            
            if self.config_path:
                cmd.extend(["--config", str(self.config_path)])
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            stdout, stderr = process.communicate(input=text.encode("utf-8"))
            
            if process.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr.decode()}")
            
            # Конвертируем bytes в numpy array
            audio = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Piper synthesis error: {e}", exc_info=True)
            raise

    def synthesize_streaming(
        self, text: str, chunk_size_samples: int = 3200
    ) -> Generator[np.ndarray, None, None]:
        """
        Стриминговый синтез (эмуляция через чанки).
        
        Args:
            text: Текст
            chunk_size_samples: Размер чанка в сэмплах
            
        Yields:
            Аудио чанки
        """
        audio = self.synthesize(text)
        
        # Разбиваем на чанки
        for i in range(0, len(audio), chunk_size_samples):
            yield audio[i : i + chunk_size_samples]


class KokoroTTS:
    """
    Основной TTS с Kokoro-82M.
    
    Примечание: Это заглушка. Реальная интеграция требует
    установки Kokoro через ONNX Runtime или PyTorch.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        sample_rate: int = 16000,
        speed: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация Kokoro TTS.
        
        Args:
            model_path: Путь к модели
            device: Устройство ('cuda' или 'cpu')
            sample_rate: Частота дискретизации
            speed: Скорость речи
            logger: Logger
        """
        self.model_path = Path(model_path)
        self.device = device
        self.sample_rate = sample_rate
        self.speed = speed
        self.logger = logger or logging.getLogger(__name__)
        
        # TODO: Загрузка модели Kokoro-82M через ONNX Runtime
        # Пока заглушка - используем Piper как основной
        self.logger.warning(
            "Kokoro-82M не реализован. Используйте Piper как основной TTS."
        )
        self._initialized = False

    def synthesize(self, text: str) -> np.ndarray:
        """
        Синтезирует речь.
        
        Args:
            text: Текст
            
        Returns:
            Аудио
        """
        if not self._initialized:
            raise NotImplementedError("Kokoro-82M TTS not implemented yet")
        
        # TODO: Реализация синтеза через Kokoro
        pass

    def synthesize_streaming(
        self, text: str, chunk_size_samples: int = 3200
    ) -> Generator[np.ndarray, None, None]:
        """Стриминговый синтез."""
        if not self._initialized:
            raise NotImplementedError("Kokoro-82M TTS not implemented yet")
        
        # TODO: Реализация стриминга
        pass


class TTSEngine:
    """Unified TTS engine с primary + fallback."""

    def __init__(
        self,
        primary_tts: Optional[KokoroTTS] = None,
        fallback_tts: Optional[PiperTTS] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация TTS engine.
        
        Args:
            primary_tts: Основной TTS (Kokoro)
            fallback_tts: Fallback TTS (Piper)
            logger: Logger
        """
        self.primary_tts = primary_tts
        self.fallback_tts = fallback_tts
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.fallback_tts:
            raise ValueError("At least fallback_tts must be provided")

    def synthesize(self, text: str, use_fallback: bool = False) -> np.ndarray:
        """
        Синтезирует речь, используя primary или fallback.
        
        Args:
            text: Текст
            use_fallback: Принудительно использовать fallback
            
        Returns:
            Аудио
        """
        if use_fallback or not self.primary_tts:
            self.logger.debug("Using fallback TTS (Piper)")
            return self.fallback_tts.synthesize(text)
        
        try:
            self.logger.debug("Using primary TTS (Kokoro)")
            return self.primary_tts.synthesize(text)
        except (NotImplementedError, Exception) as e:
            self.logger.warning(f"Primary TTS failed, falling back to Piper: {e}")
            return self.fallback_tts.synthesize(text)

    def synthesize_streaming(
        self, text: str, chunk_size_samples: int = 3200, use_fallback: bool = False
    ) -> Generator[np.ndarray, None, None]:
        """
        Стриминговый синтез.
        
        Args:
            text: Текст
            chunk_size_samples: Размер чанка
            use_fallback: Использовать fallback
            
        Yields:
            Аудио чанки
        """
        if use_fallback or not self.primary_tts:
            yield from self.fallback_tts.synthesize_streaming(text, chunk_size_samples)
        else:
            try:
                yield from self.primary_tts.synthesize_streaming(text, chunk_size_samples)
            except (NotImplementedError, Exception) as e:
                self.logger.warning(f"Primary TTS streaming failed, using fallback: {e}")
                yield from self.fallback_tts.synthesize_streaming(text, chunk_size_samples)

