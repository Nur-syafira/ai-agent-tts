#!/usr/bin/env python3

"""
Diagnostics CLI - проверяет здоровье всех сервисов и окружения.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import httpx
import os
from dotenv import load_dotenv
from src.shared.health import HealthChecker

load_dotenv()


SERVICES = {
    "ASR Gateway": os.getenv("ASR_GATEWAY_HOST", "localhost") + ":" + os.getenv("ASR_GATEWAY_PORT", "8001"),
    "LLM Service": "localhost:8000",
    "TTS Gateway": os.getenv("TTS_GATEWAY_HOST", "localhost") + ":" + os.getenv("TTS_GATEWAY_PORT", "8002"),
    "Policy Engine": os.getenv("POLICY_ENGINE_HOST", "localhost") + ":" + os.getenv("POLICY_ENGINE_PORT", "8003"),
}


async def check_service(name: str, host_port: str) -> dict:
    """Проверяет здоровье одного сервиса."""
    url = f"http://{host_port}/health"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                return {"status": "✅ healthy", "details": response.json()}
            else:
                return {"status": f"⚠️  unhealthy (HTTP {response.status_code})", "details": None}
                
    except httpx.ConnectError:
        return {"status": "❌ not running", "details": None}
    except Exception as e:
        return {"status": f"❌ error: {e}", "details": None}


async def check_redis():
    """Проверяет Redis."""
    try:
        import redis.asyncio as aioredis
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        client = await aioredis.from_url(redis_url)
        await client.ping()
        await client.close()
        
        return {"status": "✅ running", "url": redis_url}
        
    except Exception as e:
        return {"status": f"❌ error: {e}", "url": None}


def check_environment():
    """Проверяет окружение."""
    checks = {}
    
    # Python version
    checks["Python"] = f"✅ {sys.version.split()[0]}" if sys.version_info >= (3, 12) else f"⚠️  {sys.version.split()[0]} (need 3.12+)"
    
    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            checks["CUDA"] = f"✅ available ({torch.cuda.get_device_name(0)})"
        else:
            checks["CUDA"] = "❌ not available"
    except ImportError:
        checks["CUDA"] = "⚠️  PyTorch not installed"
    
    # Credentials
    creds_path = Path("credentials/google_credentials.json")
    checks["Google Credentials"] = "✅ found" if creds_path.exists() else "❌ not found"
    
    # .env
    env_path = Path(".env")
    checks[".env"] = "✅ found" if env_path.exists() else "⚠️  not found (using defaults)"
    
    return checks


async def main():
    """Главная функция."""
    print("=" * 70)
    print(" " * 20 + "Sales Agent - Health Check")
    print("=" * 70)
    print()
    
    # Проверка окружения
    print("🔍 Environment:")
    env_checks = check_environment()
    for name, status in env_checks.items():
        print(f"   {name:.<30} {status}")
    print()
    
    # Проверка GPU
    print("🔍 GPU:")
    try:
        gpu_info = HealthChecker.check_cuda_available()
        for gpu in gpu_info["gpus"]:
            print(f"   GPU {gpu['id']}: {gpu['name']}")
            print(f"      VRAM: {gpu['memory_used_mb']:.0f} / {gpu['memory_total_mb']:.0f} MB ({gpu['memory_util_percent']:.1f}%)")
            print(f"      Util: {gpu['gpu_util_percent']:.1f}%")
            print(f"      Temp: {gpu['temperature_c']}°C")
    except RuntimeError as e:
        print(f"   ⚠️  {e}")
    print()
    
    # Проверка System Stats
    print("🔍 System:")
    stats = HealthChecker.get_system_stats()
    print(f"   CPU Usage:........... {stats['cpu_percent']:.1f}%")
    print(f"   RAM Usage:........... {stats['memory_percent']:.1f}%")
    print(f"   RAM Available:....... {stats['memory_available_gb']:.2f} GB")
    print()
    
    # Проверка Redis
    print("🔍 Redis:")
    redis_status = await check_redis()
    print(f"   Status:.............. {redis_status['status']}")
    if redis_status['url']:
        print(f"   URL:................. {redis_status['url']}")
    print()
    
    # Проверка сервисов
    print("🔍 Services:")
    for name, host_port in SERVICES.items():
        result = await check_service(name, host_port)
        print(f"   {name:.<25} {result['status']}")
    print()
    
    print("=" * 70)
    print("✅ Health check completed!")
    print("=" * 70)
    print()
    print("Tip: If services are not running, start them with:")
    print("   ./venv/bin/python src/asr_gateway/main.py")
    print("   ./venv/bin/python src/tts_gateway/main.py")
    print("   ./venv/bin/python src/policy_engine/main.py")


if __name__ == "__main__":
    asyncio.run(main())

