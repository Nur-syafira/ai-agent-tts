"""
Загрузка и валидация конфигурации из YAML с возможностью переопределения через ENV.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, ValidationError


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Загружает YAML конфигурацию.
    
    Args:
        config_path: Путь к config.yaml
        
    Returns:
        Словарь с конфигурацией
        
    Raises:
        FileNotFoundError: Если файл не найден
        yaml.YAMLError: Если ошибка парсинга YAML
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def override_with_env(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Переопределяет значения конфигурации из ENV переменных.
    
    ENV переменные должны иметь формат: PREFIX_KEY_SUBKEY
    Например: ASR_GATEWAY_MODEL_NAME переопределит config["model"]["name"]
    
    Args:
        config: Словарь конфигурации
        prefix: Префикс для ENV переменных
        
    Returns:
        Обновлённая конфигурация
    """
    updated_config = config.copy()

    for key, value in config.items():
        env_key = f"{prefix}_{key}".upper() if prefix else key.upper()

        # Если значение - словарь, рекурсивно обрабатываем
        if isinstance(value, dict):
            updated_config[key] = override_with_env(value, env_key)
        else:
            # Проверяем ENV переменную
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Пытаемся сохранить тип данных
                if isinstance(value, bool):
                    updated_config[key] = env_value.lower() in ("true", "1", "yes")
                elif isinstance(value, int):
                    updated_config[key] = int(env_value)
                elif isinstance(value, float):
                    updated_config[key] = float(env_value)
                else:
                    updated_config[key] = env_value

    return updated_config


def load_and_validate_config(
    config_path: Path,
    model_class: type[BaseModel],
    env_prefix: str = "",
) -> BaseModel:
    """
    Загружает YAML конфигурацию, переопределяет из ENV и валидирует через Pydantic.
    
    Args:
        config_path: Путь к config.yaml
        model_class: Pydantic модель для валидации
        env_prefix: Префикс для ENV переменных
        
    Returns:
        Валидированная конфигурация
        
    Raises:
        ValidationError: Если конфигурация не прошла валидацию
        FileNotFoundError: Если файл не найден
    """
    # Загружаем YAML
    config_dict = load_yaml_config(config_path)

    # Переопределяем из ENV
    config_dict = override_with_env(config_dict, env_prefix)

    # Валидируем через Pydantic
    try:
        config = model_class(**config_dict)
    except ValidationError as e:
        raise ValidationError(f"Configuration validation failed: {e}") from e

    return config


def get_env_or_fail(key: str, error_message: Optional[str] = None) -> str:
    """
    Получает значение ENV переменной или выбрасывает ошибку.
    
    Args:
        key: Имя переменной
        error_message: Кастомное сообщение об ошибке
        
    Returns:
        Значение переменной
        
    Raises:
        EnvironmentError: Если переменная не установлена
    """
    value = os.getenv(key)
    if value is None:
        msg = error_message or f"Required environment variable {key} is not set"
        raise EnvironmentError(msg)
    return value

