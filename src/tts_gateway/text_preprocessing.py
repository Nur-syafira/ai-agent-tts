"""
Модуль предобработки текста для TTS.
Включает автоматическую расстановку ударений через ruaccent.
"""

import hashlib
import logging
from typing import Optional, Dict
from functools import lru_cache

try:
    from ruaccent import RUAccent
except ImportError:
    RUAccent = None


class TextPreprocessor:
    """
    Предобработка текста для TTS синтеза.
    
    Основная функция - автоматическая расстановка ударений для русского языка.
    """

    def __init__(
        self,
        use_stress_marks: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация предобработчика.
        
        Args:
            use_stress_marks: Использовать расстановку ударений
            logger: Logger
        """
        self.use_stress_marks = use_stress_marks
        self.logger = logger or logging.getLogger(__name__)
        
        # Кэш для обработанного текста
        self._cache: Dict[str, str] = {}
        
        # Инициализация ruaccent
        self.accent = None
        if use_stress_marks:
            if RUAccent is None:
                self.logger.warning(
                    "ruaccent not installed. Install with: pip install ruaccent"
                )
                self.use_stress_marks = False
            else:
                try:
                    self.accent = RUAccent()
                    # Загружаем модель при первом использовании (ленивая загрузка)
                    self.logger.info("RUAccent initialized (model will load on first use)")
                except Exception as e:
                    self.logger.error(f"Failed to initialize RUAccent: {e}")
                    self.use_stress_marks = False
                    self.accent = None

    def add_stress_marks(self, text: str) -> str:
        """
        Добавляет ударения к русскому тексту.
        
        Args:
            text: Исходный текст
            
        Returns:
            Текст с расставленными ударениями
        """
        if not self.use_stress_marks or not self.accent:
            return text
        
        # Проверяем кэш
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            self.logger.debug(f"Cache HIT for stress marks: {text[:30]}...")
            return self._cache[cache_key]
        
        try:
            # Расставляем ударения
            processed_text = self.accent.process_all(text)
            
            # Сохраняем в кэш
            self._cache[cache_key] = processed_text
            
            self.logger.debug(f"Added stress marks: {text[:30]}... -> {processed_text[:30]}...")
            return processed_text
            
        except Exception as e:
            self.logger.error(f"Error adding stress marks: {e}", exc_info=True)
            # В случае ошибки возвращаем исходный текст
            return text

    def preprocess(self, text: str, language: str = "ru") -> str:
        """
        Предобрабатывает текст перед синтезом.
        
        Args:
            text: Исходный текст
            language: Язык текста ('ru' для русского, 'en' для английского)
            
        Returns:
            Обработанный текст
        """
        processed = text
        
        # Для русского языка добавляем ударения
        if language == "ru" and self.use_stress_marks:
            processed = self.add_stress_marks(processed)
        
        return processed

    def clear_cache(self):
        """Очищает кэш обработанного текста."""
        self._cache.clear()
        self.logger.debug("Text preprocessing cache cleared")


# Глобальный экземпляр для удобства
_preprocessor: Optional[TextPreprocessor] = None


def get_preprocessor(
    use_stress_marks: bool = True,
    logger: Optional[logging.Logger] = None,
) -> TextPreprocessor:
    """
    Получает глобальный экземпляр предобработчика.
    
    Args:
        use_stress_marks: Использовать расстановку ударений
        logger: Logger
        
    Returns:
        TextPreprocessor экземпляр
    """
    global _preprocessor
    
    if _preprocessor is None:
        _preprocessor = TextPreprocessor(
            use_stress_marks=use_stress_marks,
            logger=logger,
        )
    
    return _preprocessor


def add_stress_marks(text: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Удобная функция для добавления ударений.
    
    Args:
        text: Исходный текст
        logger: Logger
        
    Returns:
        Текст с расставленными ударениями
    """
    preprocessor = get_preprocessor(logger=logger)
    return preprocessor.add_stress_marks(text)

