"""
Streaming TTS с F5-TTS для русского языка.
"""

import numpy as np
from typing import Optional, Generator, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.tts_gateway.f5_tts_engine import F5TTSEngine


class TTSEngine:
    """Unified TTS engine с F5-TTS для русского языка."""

    def __init__(
        self,
        f5_tts: Optional["F5TTSEngine"] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация TTS engine.
        
        Args:
            f5_tts: F5-TTS engine для русского языка
            logger: Logger
        """
        self.f5_tts = f5_tts
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.f5_tts:
            raise ValueError("F5-TTS engine must be provided")

    def synthesize(self, text: str, use_fallback: bool = False) -> np.ndarray:
        """
        Синтезирует речь используя F5-TTS.
        
        Args:
            text: Текст для синтеза (русский язык)
            use_fallback: Не используется (оставлено для совместимости API)
            
        Returns:
            Аудио (24 kHz, mono, float32)
        """
        try:
            self.logger.debug("Using F5-TTS for Russian")
            return self.f5_tts.synthesize(text)
        except Exception as e:
            self.logger.error(f"F5-TTS synthesis failed: {e}", exc_info=True)
            raise

    def synthesize_streaming(
        self, text: str, chunk_size_samples: int = 4800, use_fallback: bool = False
    ) -> Generator[np.ndarray, None, None]:
        """
        Стриминговый синтез с оптимизацией для меньшей задержки первого аудио.
        
        Args:
            text: Текст для синтеза (русский язык)
            chunk_size_samples: Размер чанка в сэмплах (оптимизирован для меньшей задержки)
            use_fallback: Не используется (оставлено для совместимости API)
            
        Yields:
            Аудио чанки
        """
        try:
            # Используем оптимизированный streaming для F5-TTS
            yield from self.f5_tts.synthesize_streaming(text, chunk_size_samples)
        except Exception as e:
            self.logger.error(f"F5-TTS streaming failed: {e}", exc_info=True)
            raise
