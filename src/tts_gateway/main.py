"""
TTS Gateway - FastAPI сервер для синтеза речи.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvloop
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np
import os
from dotenv import load_dotenv

from src.shared.logging_config import setup_logging
from src.shared.config_loader import load_and_validate_config
from src.shared.health import HealthChecker
from src.shared.metrics import TelemetryManager
from src.tts_gateway.streaming import TTSEngine, PiperTTS, KokoroTTS
from src.tts_gateway.prerender import PrerenderCache

load_dotenv()

logger = setup_logging("tts_gateway")


class TTSConfig(BaseModel):
    """Pydantic модель для конфигурации TTS."""
    
    class PrimaryTTSConfig(BaseModel):
        enabled: bool
        model_name: str
        model_path: str
        device: str
        sample_rate: int
        speed: float
    
    class FallbackTTSConfig(BaseModel):
        enabled: bool
        model_name: str
        model_path: str
        config_path: str
        sample_rate: int
        speed: float
    
    class PrerenderConfig(BaseModel):
        enabled: bool
        cache_dir: str
        ttl_seconds: int
        common_phrases: list[str]
    
    class StreamingConfig(BaseModel):
        chunk_size_ms: int
        buffer_ms: int
    
    class ServerConfig(BaseModel):
        host: str
        port: int
        max_connections: int
    
    class PerformanceConfig(BaseModel):
        require_cuda: bool
        use_redis_cache: bool
    
    primary_tts: PrimaryTTSConfig
    fallback_tts: FallbackTTSConfig
    prerender: PrerenderConfig
    streaming: StreamingConfig
    server: ServerConfig
    performance: PerformanceConfig


# Глобальные переменные
config: TTSConfig
tts_engine: TTSEngine
prerender_cache: PrerenderCache
telemetry: TelemetryManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    global config, tts_engine, prerender_cache, telemetry
    
    logger.info("Starting TTS Gateway...")
    
    try:
        # Загружаем конфигурацию
        config_path = Path(__file__).parent / "config.yaml"
        config = load_and_validate_config(config_path, TTSConfig, "TTS_GATEWAY")
        
        logger.info("Configuration loaded")
        
        # Инициализация telemetry
        telemetry = TelemetryManager("tts_gateway")
        
        # Инициализация fallback TTS (Piper)
        fallback_tts = None
        if config.fallback_tts.enabled:
            fallback_tts = PiperTTS(
                model_path=config.fallback_tts.model_path,
                config_path=config.fallback_tts.config_path,
                sample_rate=config.fallback_tts.sample_rate,
                speed=config.fallback_tts.speed,
                logger=logger,
            )
            logger.info("Fallback TTS (Piper) initialized")
        
        # Инициализация primary TTS (Kokoro) - опционально
        primary_tts = None
        if config.primary_tts.enabled:
            try:
                primary_tts = KokoroTTS(
                    model_path=config.primary_tts.model_path,
                    device=config.primary_tts.device,
                    sample_rate=config.primary_tts.sample_rate,
                    speed=config.primary_tts.speed,
                    logger=logger,
                )
                logger.info("Primary TTS (Kokoro) initialized")
            except NotImplementedError:
                logger.warning("Kokoro not implemented, using Piper as primary")
        
        # Создаём unified engine
        tts_engine = TTSEngine(
            primary_tts=primary_tts,
            fallback_tts=fallback_tts,
            logger=logger,
        )
        
        # Инициализация prerender cache
        prerender_cache = PrerenderCache(
            cache_dir=config.prerender.cache_dir,
            use_redis=config.performance.use_redis_cache,
            ttl_seconds=config.prerender.ttl_seconds,
            logger=logger,
        )
        
        # Пререндер частых фраз
        if config.prerender.enabled and config.prerender.common_phrases:
            prerender_cache.prerender(config.prerender.common_phrases, tts_engine)
        
        logger.info("TTS Gateway started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start TTS Gateway: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down TTS Gateway...")


# FastAPI app
app = FastAPI(
    title="TTS Gateway",
    description="Text-to-Speech service with Kokoro-82M and Piper",
    version="0.1.0",
    lifespan=lifespan,
)

health_checker = HealthChecker("tts_gateway", "0.1.0")
app.include_router(health_checker.create_router())


class SynthesizeRequest(BaseModel):
    """Запрос на синтез речи."""
    text: str
    use_fallback: bool = False


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Синтезирует речь из текста.
    
    Returns:
        Audio PCM (16 kHz, mono, float32)
    """
    try:
        # Проверяем кэш
        cached_audio = prerender_cache.get(request.text)
        
        if cached_audio is not None:
            logger.info(f"Serving from cache: {request.text[:30]}...")
        else:
            # Синтезируем
            with telemetry.trace_span("tts_synthesize"):
                cached_audio = tts_engine.synthesize(
                    request.text, use_fallback=request.use_fallback
                )
            
            # Кэшируем
            prerender_cache.set(request.text, cached_audio)
        
        # Конвертируем в bytes
        audio_bytes = cached_audio.astype(np.float32).tobytes()
        
        return Response(
            content=audio_bytes,
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(config.fallback_tts.sample_rate),
                "X-Channels": "1",
                "X-Format": "float32",
            },
        )
    
    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "tts_gateway", "status": "running"}


def main():
    """Точка входа."""
    uvloop.install()
    
    import uvicorn
    
    host = os.getenv("TTS_GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("TTS_GATEWAY_PORT", 8002))
    
    uvicorn.run(
        "src.tts_gateway.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()

