"""
Структурированное JSON-логирование для всех сервисов.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
import os


class JSONFormatter(logging.Formatter):
    """Форматтер для структурированных JSON-логов."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Форматирует log record в JSON.
        
        Args:
            record: Запись лога
            
        Returns:
            JSON-строка с логом
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": record.name,
            "message": record.getMessage(),
        }

        # Добавляем метаданные если есть
        if hasattr(record, "extra"):
            log_data["meta"] = record.extra

        # Добавляем информацию об ошибке если есть
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Добавляем контекст
        if hasattr(record, "context"):
            log_data["context"] = record.context

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(service_name: str, level: str = None) -> logging.Logger:
    """
    Настраивает логирование для сервиса.
    
    Args:
        service_name: Имя сервиса (например, 'asr_gateway')
        level: Уровень логирования (по умолчанию из ENV LOG_LEVEL или INFO)
        
    Returns:
        Настроенный logger
        
    Raises:
        ValueError: Если уровень логирования некорректный
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "json")

    # Создаём logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []  # Очищаем существующие handlers

    # Создаём handler для stdout
    handler = logging.StreamHandler(sys.stdout)

    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        # Простой текстовый формат для разработки
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Отключаем propagation чтобы не дублировать логи
    logger.propagate = False

    logger.info(f"{service_name} logging initialized", extra={"level": log_level})

    return logger


def log_with_context(logger: logging.Logger, level: str, message: str, **context):
    """
    Логирование с дополнительным контекстом.
    
    Args:
        logger: Logger объект
        level: Уровень логирования ('info', 'warning', 'error', etc.)
        message: Сообщение
        **context: Дополнительные поля контекста
    """
    log_func = getattr(logger, level.lower())
    
    # Создаём LogRecord с контекстом
    extra = {"context": context}
    log_func(message, extra=extra)

