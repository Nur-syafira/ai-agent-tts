#!/usr/bin/env python3
"""Проверка готовности vLLM."""

import sys
import time
import httpx


def check_vllm(url: str = "http://localhost:8000/v1/models", timeout: int = 300):
    """
    Проверяет готовность vLLM сервера.
    
    Args:
        url: URL для проверки
        timeout: Максимальное время ожидания (сек)
    
    Returns:
        True если сервер готов, False иначе
    """
    print(f"🔍 Проверка vLLM на {url}...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    if models:
                        model_id = models[0].get("id", "unknown")
                        print(f"\n✅ vLLM готов!")
                        print(f"   Модель: {model_id}")
                        print(f"   Время загрузки: {int(time.time() - start)}с")
                        return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        
        elapsed = int(time.time() - start)
        print(f"⏳ Ожидание... ({elapsed}s / {timeout}s)", end="\r", flush=True)
        time.sleep(5)
    
    print(f"\n❌ vLLM не запустился за {timeout}с")
    return False


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1/models"
    success = check_vllm(url)
    sys.exit(0 if success else 1)

