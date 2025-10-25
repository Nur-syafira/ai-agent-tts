"""
Health-check endpoints для сервисов.
"""

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from typing import Dict, Optional, Any
from datetime import datetime
import psutil
import GPUtil


class HealthStatus(BaseModel):
    """Модель статуса здоровья сервиса."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    service_name: str
    version: str
    uptime_seconds: float
    checks: Dict[str, bool]
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """Класс для проверки здоровья сервиса."""

    def __init__(self, service_name: str, version: str = "0.1.0"):
        """
        Инициализация health checker.
        
        Args:
            service_name: Имя сервиса
            version: Версия сервиса
        """
        self.service_name = service_name
        self.version = version
        self.start_time = datetime.utcnow()

    def create_router(self) -> APIRouter:
        """
        Создаёт FastAPI router с health endpoints.
        
        Returns:
            APIRouter с /health и /ready endpoints
        """
        router = APIRouter(tags=["health"])

        @router.get("/health")
        async def health_check(response: Response):
            """Базовая проверка здоровья сервиса."""
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            health_status = HealthStatus(
                status="healthy",
                timestamp=datetime.utcnow().isoformat() + "Z",
                service_name=self.service_name,
                version=self.version,
                uptime_seconds=uptime,
                checks={"service": True},
            )
            
            return health_status.dict()

        @router.get("/ready")
        async def readiness_check(response: Response):
            """Проверка готовности сервиса к обработке запросов."""
            is_ready = await self._check_readiness()
            
            if not is_ready:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                
            return {"ready": is_ready, "timestamp": datetime.utcnow().isoformat() + "Z"}

        return router

    async def _check_readiness(self) -> bool:
        """
        Проверка готовности (переопределяется в наследниках).
        
        Returns:
            True если сервис готов
        """
        return True

    @staticmethod
    def check_cuda_available() -> Dict[str, Any]:
        """
        Проверка наличия и состояния CUDA GPU.
        
        Returns:
            Словарь с информацией о GPU
            
        Raises:
            RuntimeError: Если CUDA недоступна
        """
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                raise RuntimeError("No CUDA GPUs detected")

            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_free_mb": gpu.memoryFree,
                    "memory_util_percent": gpu.memoryUtil * 100,
                    "gpu_util_percent": gpu.load * 100,
                    "temperature_c": gpu.temperature,
                })

            return {
                "available": True,
                "gpu_count": len(gpus),
                "gpus": gpu_info,
            }

        except Exception as e:
            raise RuntimeError(f"CUDA check failed: {e}") from e

    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """
        Получает статистику системы.
        
        Returns:
            Словарь с CPU, RAM, GPU статистикой
        """
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        }

        try:
            gpu_stats = HealthChecker.check_cuda_available()
            stats["gpu"] = gpu_stats
        except RuntimeError:
            stats["gpu"] = {"available": False}

        return stats

