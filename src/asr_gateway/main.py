"""
ASR Gateway - FastAPI сервер для потокового распознавания речи.
"""

import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import uvloop
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import numpy as np
import os
from dotenv import load_dotenv

from src.shared.logging_config import setup_logging
from src.shared.config_loader import load_and_validate_config
from src.shared.health import HealthChecker
from src.shared.metrics import TelemetryManager
from src.asr_gateway.streaming import StreamingASR
from pydantic import BaseModel


# Загружаем .env
load_dotenv()

# Настраиваем логирование
logger = setup_logging("asr_gateway")


class ASRConfig(BaseModel):
    """Pydantic модель для конфигурации ASR."""
    
    class ModelConfig(BaseModel):
        name: str
        device: str
        compute_type: str
        beam_size: int
        language: str
    
    class VADConfig(BaseModel):
        enabled: bool
        model: str
        threshold: float
        min_speech_duration_ms: int
        max_speech_duration_s: int
        min_silence_duration_ms: int
        speech_pad_ms: int
    
    class StreamingConfig(BaseModel):
        enable_realtime_transcription: bool
        partial_transcript_interval_ms: int
        audio_chunk_ms: int
        sample_rate: int
        channels: int
    
    class ServerConfig(BaseModel):
        host: str
        port: int
        max_connections: int
    
    class PerformanceConfig(BaseModel):
        require_cuda: bool
        min_vram_mb: int
        warmup: bool
    
    model: ModelConfig
    vad: VADConfig
    streaming: StreamingConfig
    server: ServerConfig
    performance: PerformanceConfig


# Глобальные переменные
config: ASRConfig
asr_engine: StreamingASR
telemetry: TelemetryManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager для инициализации и cleanup."""
    global config, asr_engine, telemetry
    
    logger.info("Starting ASR Gateway...")
    
    try:
        # Загружаем конфигурацию
        config_path = Path(__file__).parent / "config.yaml"
        config = load_and_validate_config(config_path, ASRConfig, "ASR_GATEWAY")
        
        logger.info("Configuration loaded", extra={"context": config.dict()})
        
        # Guard-проверка CUDA
        if config.performance.require_cuda:
            cuda_info = HealthChecker.check_cuda_available()
            logger.info("CUDA check passed", extra={"context": cuda_info})
            
            # Проверяем VRAM
            total_vram = cuda_info["gpus"][0]["memory_total_mb"]
            if total_vram < config.performance.min_vram_mb:
                raise RuntimeError(
                    f"Insufficient VRAM: {total_vram} MB < {config.performance.min_vram_mb} MB"
                )
        
        # Инициализация telemetry
        telemetry = TelemetryManager("asr_gateway")
        
        # Инициализация ASR
        asr_engine = StreamingASR(
            model_name=config.model.name,
            device=config.model.device,
            compute_type=config.model.compute_type,
            language=config.model.language,
            beam_size=config.model.beam_size,
            vad_enabled=config.vad.enabled,
            vad_threshold=config.vad.threshold,
            min_speech_duration_ms=config.vad.min_speech_duration_ms,
            max_speech_duration_s=config.vad.max_speech_duration_s,
            min_silence_duration_ms=config.vad.min_silence_duration_ms,
            logger=logger,
        )
        
        # Warmup
        if config.performance.warmup:
            asr_engine.warmup()
        
        logger.info("ASR Gateway started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start ASR Gateway: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down ASR Gateway...")
        if asr_engine:
            asr_engine.shutdown()


# Создаём FastAPI приложение
app = FastAPI(
    title="ASR Gateway",
    description="Streaming ASR service with RealtimeSTT and Silero VAD",
    version="0.1.0",
    lifespan=lifespan,
)

# Добавляем health endpoints
health_checker = HealthChecker("asr_gateway", "0.1.0")
app.include_router(health_checker.create_router())


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint для потокового распознавания.
    
    Клиент отправляет PCM audio chunks (16 kHz, mono, float32).
    Сервер возвращает JSON с partial и final транскриптами.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Получаем аудио данные от клиента
            data = await websocket.receive_bytes()
            
            # Конвертируем bytes в numpy array
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Транскрибируем
            with telemetry.trace_span("asr_transcribe_chunk"):
                transcript = asr_engine.transcribe_audio_chunk(
                    audio_chunk, config.streaming.sample_rate
                )
            
            # Отправляем результат
            if transcript:
                await websocket.send_json({
                    "type": "partial",
                    "text": transcript,
                    "timestamp": asyncio.get_event_loop().time(),
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "asr_gateway", "status": "running"}


def main():
    """Точка входа приложения."""
    # Используем uvloop для лучшей производительности
    uvloop.install()
    
    import uvicorn
    
    host = os.getenv("ASR_GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("ASR_GATEWAY_PORT", 8001))
    
    uvicorn.run(
        "src.asr_gateway.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,  # Используем наш structured logging
    )


if __name__ == "__main__":
    main()

