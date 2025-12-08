"""
Загрузка и валидация конфигурации из YAML с возможностью переопределения через ENV.

Использует pydantic-settings BaseSettings с кастомным источником для YAML.
"""

import yaml
from pathlib import Path
from typing import Any, Optional, ClassVar

from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    Кастомный источник настроек для загрузки конфигурации из YAML файла.
    
    Загружает данные из YAML файла и возвращает их в виде словаря для BaseSettings.
    """

    def __init__(self, settings_cls: type[BaseSettings], yaml_file: Path):
        """
        Инициализация источника YAML настроек.
        
        Args:
            settings_cls: Класс настроек
            yaml_file: Путь к YAML файлу конфигурации
        """
        super().__init__(settings_cls)
        self.yaml_file = yaml_file

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        """
        Получает значение поля из YAML файла.
        
        Args:
            field: Информация о поле
            field_name: Имя поля
            
        Returns:
            Кортеж (значение, имя поля, является ли значение сложным)
        """
        if not self.yaml_file.exists():
            return None, field_name, False

        encoding = self.config.get("env_file_encoding", "utf-8")
        try:
            with open(self.yaml_file, "r", encoding=encoding) as f:
                yaml_content = yaml.safe_load(f)
        except Exception:
            return None, field_name, False

        if yaml_content is None:
            return None, field_name, False

        # Поддержка вложенных структур
        field_value = yaml_content.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        """
        Подготавливает значение поля для использования.
        
        Args:
            field_name: Имя поля
            field: Информация о поле
            value: Значение из YAML
            value_is_complex: Является ли значение сложным
            
        Returns:
            Подготовленное значение
        """
        return value

    def __call__(self) -> dict[str, Any]:
        """
        Загружает все значения из YAML файла.
        
        Returns:
            Словарь с настройками из YAML файла
        """
        d: dict[str, Any] = {}

        if not self.yaml_file.exists():
            return d

        encoding = self.config.get("env_file_encoding", "utf-8")
        try:
            with open(self.yaml_file, "r", encoding=encoding) as f:
                yaml_content = yaml.safe_load(f)
        except Exception:
            return d

        if yaml_content is None:
            return d

        # Загружаем все поля из YAML
        for field_name, field in self.settings_cls.model_fields.items():
            field_value = yaml_content.get(field_name)
            if field_value is not None:
                field_value = self.prepare_field_value(
                    field_name,
                    field,
                    field_value,
                    False,
                )
                if field_value is not None:
                    d[field_name] = field_value

        return d


class YamlBaseSettings(BaseSettings):
    """
    Базовый класс для настроек с поддержкой YAML файлов.
    
    Использование:
        class MyConfig(YamlBaseSettings):
            model_config = SettingsConfigDict(env_prefix="MY_")
            yaml_file = Path("config.yaml")
            
            field1: str
            field2: int
    """

    yaml_file: Path | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Кастомизирует источники настроек с добавлением YAML источника.
        
        Приоритет: init_settings > env_settings > yaml_file > dotenv_settings
        """
        # Получаем путь к YAML файлу из класса или из model_config
        yaml_file = getattr(cls, "yaml_file", None)
        if yaml_file is None:
            # Пытаемся получить из model_config
            model_config = getattr(cls, "model_config", None)
            if model_config and hasattr(model_config, "yaml_file"):
                yaml_file = model_config.yaml_file

        if yaml_file is None:
            # Если YAML файл не указан, используем стандартные источники
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
            )

        yaml_source = YamlConfigSettingsSource(settings_cls, Path(yaml_file))
        return (
            init_settings,
            env_settings,
            yaml_source,
            dotenv_settings,
            file_secret_settings,
        )


def load_and_validate_config(
    config_path: Path,
    model_class: type[BaseSettings],
    env_prefix: str = "",
) -> BaseSettings:
    """
    Загружает YAML конфигурацию, переопределяет из ENV и валидирует через Pydantic.
    
    Использует BaseSettings с кастомным источником для YAML.
    Приоритет источников:
    1. init_settings (аргументы конструктора)
    2. env_settings (ENV переменные с префиксом)
    3. yaml_file (YAML файл)
    4. dotenv_settings (.env файл)
    
    Args:
        config_path: Путь к config.yaml
        model_class: Класс настроек, наследующийся от BaseSettings
        env_prefix: Префикс для ENV переменных (например, "ASR_GATEWAY")
        
    Returns:
        Валидированная конфигурация
        
    Raises:
        ValidationError: Если конфигурация не прошла валидацию
        FileNotFoundError: Если файл не найден
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Создаем временный класс с YAML поддержкой
    class ConfigWithYaml(model_class):
        yaml_file: ClassVar[Path] = config_path

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            """Кастомизирует источники настроек с добавлением YAML источника."""
            yaml_source = YamlConfigSettingsSource(settings_cls, config_path)
            return (
                init_settings,
                env_settings,
                yaml_source,
                dotenv_settings,
                file_secret_settings,
            )

    # Настраиваем model_config с префиксом ENV переменных
    env_prefix_normalized = env_prefix.lower() + "_" if env_prefix else ""
    model_config = SettingsConfigDict(env_prefix=env_prefix_normalized)

    # Создаем экземпляр настроек
    config = ConfigWithYaml(model_config=model_config)

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
    import os

    value = os.getenv(key)
    if value is None:
        msg = error_message or f"Required environment variable {key} is not set"
        raise EnvironmentError(msg)
    return value
