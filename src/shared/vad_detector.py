"""
VAD Detector - прямой интерфейс к Silero VAD для barge-in detection.

Использует Silero VAD напрямую (не через RealtimeSTT) для независимого
детектирования речи на входящем аудио канале во время воспроизведения TTS.
"""

import torch
import numpy as np
from typing import Optional, Deque
from collections import deque
import logging


class VADDetector:
    """
    Детектор голосовой активности на базе Silero VAD.
    
    Используется для barge-in detection - определения момента, когда
    пользователь начинает говорить во время воспроизведения TTS агента.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 100,
        sample_rate: int = 16000,
        device: Optional[str] = None,
        use_onnx: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация VAD Detector.
        
        Args:
            threshold: Порог вероятности речи (0.0-1.0)
            min_speech_duration_ms: Минимальная длительность речи для подтверждения (мс)
            sample_rate: Частота дискретизации аудио (Hz)
            device: Устройство ('cuda', 'cpu' или None для автоопределения)
            use_onnx: Использовать ONNX версию модели (быстрее)
            logger: Logger объект
            
        Raises:
            RuntimeError: Если не удалось загрузить модель Silero VAD
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.sample_rate = sample_rate
        self.logger = logger or logging.getLogger(__name__)
        
        # Определяем устройство
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Загружаем модель Silero VAD
        try:
            self.logger.info(
                f"Loading Silero VAD model (device={self.device}, onnx={use_onnx})"
            )
            
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=use_onnx,
            )
            
            if not use_onnx:
                self.model = self.model.to(self.device)
                self.model.eval()
            
            self.logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Silero VAD model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Silero VAD model: {e}") from e
        
        # Буфер для отслеживания непрерывной речи
        self.speech_buffer: Deque[bool] = deque(maxlen=int(
            self.min_speech_duration_ms / (1000.0 / self.sample_rate * 512)  # Примерно количество чанков
        ))
        
        # Состояние
        self.is_speech_detected = False
        self.consecutive_speech_chunks = 0
        
    def detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Определяет наличие речи в аудио чанке.
        
        Args:
            audio_chunk: Аудио данные (float32, нормализованные в [-1, 1])
                         Длина должна быть кратна 512 сэмплам для Silero VAD
            
        Returns:
            True если обнаружена речь, False иначе
        """
        try:
            # Проверяем размер чанка
            if len(audio_chunk) < 512:
                # Если чанк слишком маленький, накапливаем в буфере
                # Для простоты возвращаем False для маленьких чанков
                return False
            
            # Конвертируем в torch tensor
            if isinstance(audio_chunk, np.ndarray):
                audio_tensor = torch.from_numpy(audio_chunk).float()
            else:
                audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
            
            # Перемещаем на нужное устройство
            audio_tensor = audio_tensor.to(self.device)
            
            # Silero VAD ожидает аудио длиной кратное 512 сэмплам
            # Обрезаем до ближайшего кратного 512
            num_samples = len(audio_tensor)
            num_samples_aligned = (num_samples // 512) * 512
            
            if num_samples_aligned < 512:
                return False
            
            audio_tensor = audio_tensor[:num_samples_aligned]
            
            # Получаем вероятность речи
            with torch.no_grad():
                # Silero VAD API: model(audio_tensor, sample_rate) возвращает вероятность речи
                # Модель может быть PyTorch или ONNX, но API одинаковый
                try:
                    speech_prob = self.model(audio_tensor, self.sample_rate)
                    # Результат может быть tensor или numpy array
                    if isinstance(speech_prob, torch.Tensor):
                        speech_prob = speech_prob.item()
                    elif isinstance(speech_prob, np.ndarray):
                        speech_prob = speech_prob.item() if speech_prob.size == 1 else float(speech_prob[0])
                    else:
                        speech_prob = float(speech_prob)
                except Exception as e:
                    # Если прямой вызов не работает, пробуем альтернативный способ
                    self.logger.warning(f"Direct VAD call failed, trying alternative: {e}")
                    # Для некоторых версий Silero VAD может потребоваться другой формат
                    speech_prob = 0.0
            
            # Определяем наличие речи
            is_speech = speech_prob > self.threshold
            
            # Обновляем буфер для фильтрации кратковременных срабатываний
            self.speech_buffer.append(is_speech)
            
            # Подсчитываем последовательные чанки с речью
            if is_speech:
                self.consecutive_speech_chunks += 1
            else:
                self.consecutive_speech_chunks = 0
            
            # Подтверждаем речь только если она длится достаточно долго
            min_chunks = max(1, int(
                self.min_speech_duration_ms / (len(audio_tensor) / self.sample_rate * 1000)
            ))
            
            if self.consecutive_speech_chunks >= min_chunks:
                if not self.is_speech_detected:
                    self.logger.debug(
                        f"Speech detected (prob={speech_prob:.3f}, "
                        f"chunks={self.consecutive_speech_chunks})"
                    )
                self.is_speech_detected = True
                return True
            else:
                # Если речь прервалась, сбрасываем флаг
                if self.is_speech_detected and self.consecutive_speech_chunks == 0:
                    self.logger.debug("Speech ended")
                    self.is_speech_detected = False
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error in speech detection: {e}", exc_info=True)
            return False
    
    def reset(self):
        """Сбрасывает состояние детектора."""
        self.speech_buffer.clear()
        self.is_speech_detected = False
        self.consecutive_speech_chunks = 0
        self.logger.debug("VAD detector reset")
    
    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Возвращает вероятность речи без фильтрации по длительности.
        
        Args:
            audio_chunk: Аудио данные (float32, нормализованные в [-1, 1])
            
        Returns:
            Вероятность речи (0.0-1.0)
        """
        try:
            if len(audio_chunk) < 512:
                return 0.0
            
            audio_tensor = torch.from_numpy(audio_chunk).float()
            audio_tensor = audio_tensor.to(self.device)
            
            num_samples = len(audio_tensor)
            num_samples_aligned = (num_samples // 512) * 512
            
            if num_samples_aligned < 512:
                return 0.0
            
            audio_tensor = audio_tensor[:num_samples_aligned]
            
            with torch.no_grad():
                try:
                    speech_prob = self.model(audio_tensor, self.sample_rate)
                    # Результат может быть tensor или numpy array
                    if isinstance(speech_prob, torch.Tensor):
                        speech_prob = speech_prob.item()
                    elif isinstance(speech_prob, np.ndarray):
                        speech_prob = speech_prob.item() if speech_prob.size == 1 else float(speech_prob[0])
                    else:
                        speech_prob = float(speech_prob)
                except Exception as e:
                    self.logger.warning(f"VAD call failed: {e}")
                    speech_prob = 0.0
            
            return float(speech_prob)
            
        except Exception as e:
            self.logger.error(f"Error getting speech probability: {e}", exc_info=True)
            return 0.0

