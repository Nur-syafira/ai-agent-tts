"""
LLM Service - FastAPI обёртка для vLLM с OpenAI-compatible API.

Примечание: Пользователь запускает vLLM сервер отдельно.
Этот модуль содержит helper-функции и документацию для запуска.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI

from src.shared.logging_config import setup_logging
from src.shared.config_loader import load_and_validate_config

load_dotenv()

logger = setup_logging("llm_service")


class LLMConfig(BaseSettings):
    """Pydantic Settings модель для конфигурации LLM."""
    
    class ModelConfig(BaseModel):
        name: str
        quantization: str
        max_model_len: int
        gpu_memory_utilization: float
    
    class ServerConfig(BaseModel):
        host: str
        port: int
        api_key: Optional[str] = None
    
    class GenerationConfig(BaseModel):
        temperature: float
        top_p: float
        max_tokens: int
        response_format_type: str
    
    class PerformanceConfig(BaseModel):
        enable_chunked_prefill: bool
        max_num_batched_tokens: int
        enable_prefix_caching: bool
    
    class GuardsConfig(BaseModel):
        require_cuda: bool
        min_vram_mb: int
    
    model: ModelConfig
    server: ServerConfig
    generation: GenerationConfig
    performance: PerformanceConfig
    guards: GuardsConfig


class LLMClient:
    """Клиент для взаимодействия с vLLM сервером."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Инициализация LLM клиента.
        
        Args:
            base_url: URL vLLM сервера (например, http://localhost:8000/v1)
            api_key: API ключ (опционально)
            model_name: Имя модели в vLLM (опционально, берется из конфига если не указано)
        """
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )
        self.logger = logger
        self.model_name = model_name  # Сохраняем имя модели для использования в запросах

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, str]] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Генерирует ответ от LLM.
        
        Args:
            messages: Список сообщений в формате OpenAI
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            response_format: Формат ответа (для structured output)
            model_name: Имя модели (если не указано, используется из конфига)
            
        Returns:
            Сгенерированный текст
            
        Raises:
            Exception: При ошибке генерации
        """
        try:
            # Если нужен JSON ответ
            extra_body = {}
            if response_format and response_format.get("type") == "json_object":
                extra_body["response_format"] = response_format

            # Используем переданное имя модели или дефолтное
            model = model_name or self.model_name or "models/Qwen3-16B-A3B-abliterated-AWQ"

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body if extra_body else None,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"LLM generation error: {e}", exc_info=True)
            raise

    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,  # Ниже для более детерминированного JSON
        max_tokens: int = 512,
    ) -> str:
        """
        Генерирует structured JSON ответ.
        
        Args:
            messages: Список сообщений
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            
        Returns:
            JSON строка
        """
        return await self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    async def health_check(self) -> bool:
        """
        Проверяет доступность vLLM сервера.
        
        Returns:
            True если сервер доступен
        """
        try:
            # Простой запрос для проверки
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


def get_vllm_launch_command(config: LLMConfig) -> str:
    """
    Генерирует команду для запуска vLLM сервера.
    
    Args:
        config: Конфигурация LLM
        
    Returns:
        Команда для запуска vLLM
    """
    cmd = [
        "vllm", "serve",
        config.model.name,
        f"--host {config.server.host}",
        f"--port {config.server.port}",
        f"--max-model-len {config.model.max_model_len}",
        f"--gpu-memory-utilization {config.model.gpu_memory_utilization}",
        f"--quantization {config.model.quantization}",
    ]
    
    if config.performance.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    
    if config.performance.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    
    if config.performance.max_num_batched_tokens:
        cmd.append(f"--max-num-batched-tokens {config.performance.max_num_batched_tokens}")
    
    return " \\\n  ".join(cmd)


def main():
    """
    Точка входа - выводит инструкции по запуску vLLM.
    """
    print("=" * 60)
    print("LLM Service - vLLM Server Launch Instructions")
    print("=" * 60)
    
    # Загружаем конфигурацию
    config_path = Path(__file__).parent / "config.yaml"
    config = load_and_validate_config(config_path, LLMConfig, "LLM_SERVICE")
    
    print("\n🚀 Пользователь должен запустить vLLM сервер самостоятельно.\n")
    print("Рекомендуемая команда для запуска:\n")
    
    cmd = get_vllm_launch_command(config)
    print(f"  {cmd}\n")
    
    print("\nАльтернативно, через Python API:\n")
    print("""
from vllm import LLM

llm = LLM(
    model="models/Qwen3-16B-A3B-abliterated-AWQ",  # Локальный путь к модели
    quantization="awq",
    max_model_len=2048,
    gpu_memory_utilization=0.75,
)
""")
    
    print("\n" + "=" * 60)
    print("После запуска vLLM:")
    print(f"  - API доступен на: http://{config.server.host}:{config.server.port}/v1")
    print(f"  - OpenAI-compatible endpoint: /v1/chat/completions")
    print("=" * 60)


if __name__ == "__main__":
    main()

