"""
FreeSWITCH Bridge - WebSocket сервер для интеграции с FreeSWITCH mod_audio_fork.

Обрабатывает двунаправленный аудио поток:
- Входящий аудио (от клиента) → ASR Gateway → Policy Engine → TTS Gateway → Исходящий аудио (к клиенту)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import uvloop
import base64
import json
import logging
import uuid
from typing import Optional, Dict
from contextlib import asynccontextmanager
import numpy as np
import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from src.shared.logging_config import setup_logging
from src.shared.config_loader import load_and_validate_config
from src.shared.health import HealthChecker
from src.shared.metrics import TelemetryManager
from src.shared.vad_detector import VADDetector

load_dotenv()

logger = setup_logging("freeswitch_bridge")


class FreeSWITCHConfig(BaseSettings):
    """Pydantic Settings модель для конфигурации FreeSWITCH Bridge."""
    
    class ServicesConfig(BaseModel):
        asr_ws_url: str  # WebSocket URL для ASR Gateway
        tts_http_url: str  # HTTP URL для TTS Gateway
        policy_http_url: str  # HTTP URL для Policy Engine
    
    class AudioConfig(BaseModel):
        sample_rate: int
        channels: int
        format: str  # "L16" для PCM 16-bit
        chunk_size_ms: int
    
    class BargeInConfig(BaseModel):
        enabled: bool
        vad_threshold: float
        min_speech_duration_ms: int
        use_onnx: bool = True
        device: Optional[str] = None
    
    class ServerConfig(BaseModel):
        host: str
        port: int
        max_connections: int
    
    class EndpointingConfig(BaseModel):
        base_silence_threshold_ms: float
        fast_speech_threshold_ms: float
        slow_speech_threshold_ms: float
        min_threshold_ms: float
        max_threshold_ms: float
        adaptive_enabled: bool = True
    
    services: ServicesConfig
    audio: AudioConfig
    barge_in: BargeInConfig
    endpointing: EndpointingConfig
    server: ServerConfig


# Глобальные переменные
config: FreeSWITCHConfig
telemetry: TelemetryManager
http_client: httpx.AsyncClient
barge_in_vad: Optional[VADDetector] = None


class CallSession:
    """Состояние одного звонка."""
    
    def __init__(self, call_id: str, freeswitch_ws: WebSocket):
        self.call_id = call_id
        self.freeswitch_ws = freeswitch_ws  # WebSocket соединение с FreeSWITCH
        self.session_id: Optional[str] = None  # Policy Engine session ID
        self.asr_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.current_tts_task: Optional[asyncio.Task] = None
        self.is_speaking = False  # Агент говорит
        self.last_user_speech_time = 0.0
        self.buffer: list[np.ndarray] = []  # Буфер для аудио
        
    def reset(self):
        """Сброс состояния."""
        self.is_speaking = False
        self.buffer.clear()


# Хранилище активных звонков
active_calls: Dict[str, CallSession] = {}


def decode_audio_data(base64_data: str, sample_rate: int) -> np.ndarray:
    """
    Декодирует base64 аудио данные в numpy array.
    
    Args:
        base64_data: Base64 encoded PCM audio
        sample_rate: Частота дискретизации
        
    Returns:
        Numpy array (float32, normalized to [-1, 1])
    """
    # Декодируем base64
    audio_bytes = base64.b64decode(base64_data)
    
    # Конвертируем в numpy array (L16 = int16)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Нормализуем в float32 [-1, 1]
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    return audio_float32


def encode_audio_data(audio: np.ndarray) -> str:
    """
    Кодирует numpy array в base64 для отправки в FreeSWITCH.
    
    Args:
        audio: Numpy array (float32, normalized to [-1, 1])
        
    Returns:
        Base64 encoded PCM audio (L16)
    """
    # Конвертируем float32 в int16
    audio_int16 = (audio * 32767.0).astype(np.int16)
    
    # Кодируем в base64
    audio_bytes = audio_int16.tobytes()
    base64_data = base64.b64encode(audio_bytes).decode('utf-8')
    
    return base64_data


async def connect_to_asr(call_id: str) -> Optional[websockets.WebSocketClientProtocol]:
    """
    Подключается к ASR Gateway через WebSocket.
    
    Args:
        call_id: ID звонка
        
    Returns:
        WebSocket соединение или None при ошибке
    """
    try:
        ws_url = config.services.asr_ws_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/transcribe"
        
        logger.info(f"[{call_id}] Connecting to ASR Gateway: {ws_url}")
        
        ws = await websockets.connect(ws_url)
        logger.info(f"[{call_id}] Connected to ASR Gateway")
        
        return ws
        
    except Exception as e:
        logger.error(f"[{call_id}] Failed to connect to ASR Gateway: {e}", exc_info=True)
        return None


async def send_audio_to_asr(
    ws: websockets.WebSocketClientProtocol,
    audio: np.ndarray,
    call_id: str,
):
    """
    Отправляет аудио в ASR Gateway.
    
    Args:
        ws: WebSocket соединение
        audio: Аудио данные (float32)
        call_id: ID звонка
    """
    try:
        # ASR Gateway ожидает bytes (float32)
        audio_bytes = audio.astype(np.float32).tobytes()
        await ws.send(audio_bytes)
    except Exception as e:
        logger.error(f"[{call_id}] Failed to send audio to ASR: {e}", exc_info=True)


async def receive_transcript_from_asr(
    ws: websockets.WebSocketClientProtocol,
    call_id: str,
) -> Optional[str]:
    """
    Получает транскрипт от ASR Gateway.
    
    Args:
        ws: WebSocket соединение
        call_id: ID звонка
        
    Returns:
        Транскрипт или None
    """
    try:
        message = await asyncio.wait_for(ws.recv(), timeout=0.1)
        data = json.loads(message)
        
        if data.get("type") == "partial" and data.get("text"):
            return data["text"]
        
        return None
        
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        logger.error(f"[{call_id}] Failed to receive transcript from ASR: {e}", exc_info=True)
        return None


async def synthesize_speech(text: str, call_id: str) -> Optional[np.ndarray]:
    """
    Синтезирует речь через TTS Gateway.
    
    Args:
        text: Текст для синтеза
        call_id: ID звонка
        
    Returns:
        Аудио данные или None при ошибке
    """
    try:
        url = f"{config.services.tts_http_url}/synthesize"
        
        response = await http_client.post(
            url,
            json={"text": text, "use_fallback": False},
            timeout=10.0,
        )
        
        if response.status_code == 200:
            # TTS Gateway возвращает PCM audio (float32)
            audio_bytes = response.content
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Ресемплируем если нужно
            if len(audio) > 0:
                # Проверяем sample rate из заголовков
                tts_sample_rate = int(response.headers.get("X-Sample-Rate", config.audio.sample_rate))
                if tts_sample_rate != config.audio.sample_rate:
                    import librosa
                    audio = librosa.resample(
                        audio,
                        orig_sr=tts_sample_rate,
                        target_sr=config.audio.sample_rate,
                    )
            
            logger.debug(f"[{call_id}] Synthesized speech: {len(audio)} samples")
            return audio
        
        else:
            logger.error(f"[{call_id}] TTS synthesis failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"[{call_id}] TTS synthesis error: {e}", exc_info=True)
        return None


async def process_dialog(user_message: str, call_id: str) -> Optional[str]:
    """
    Обрабатывает диалог через Policy Engine.
    
    Args:
        user_message: Сообщение пользователя
        call_id: ID звонка
        
    Returns:
        Ответ агента или None при ошибке
    """
    try:
        session = active_calls.get(call_id)
        if not session:
            logger.error(f"[{call_id}] Session not found")
            return None
        
        # Создаем или получаем session_id для Policy Engine
        if not session.session_id:
            session.session_id = f"call-{call_id}"
        
        url = f"{config.services.policy_http_url}/dialog"
        
        response = await http_client.post(
            url,
            json={
                "session_id": session.session_id,
                "user_message": user_message,
            },
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            agent_message = data.get("agent_message", "")
            is_complete = data.get("is_complete", False)
            
            logger.info(
                f"[{call_id}] Dialog processed",
                extra={
                    "context": {
                        "user_message": user_message[:50],
                        "agent_message": agent_message[:50],
                        "is_complete": is_complete,
                    }
                },
            )
            
            return agent_message
        else:
            logger.error(f"[{call_id}] Dialog processing failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"[{call_id}] Dialog processing error: {e}", exc_info=True)
        return None


async def send_audio_to_freeswitch(
    websocket: WebSocket,
    audio: np.ndarray,
    call_id: str,
):
    """
    Отправляет аудио в FreeSWITCH.
    
    Args:
        websocket: WebSocket соединение с FreeSWITCH
        audio: Аудио данные (float32)
        call_id: ID звонка
    """
    try:
        # Кодируем в base64
        base64_data = encode_audio_data(audio)
        
        # Формируем сообщение для FreeSWITCH
        message = {
            "type": "audio",
            "direction": "output",
            "format": config.audio.format,
            "sample_rate": config.audio.sample_rate,
            "channels": config.audio.channels,
            "data": base64_data,
        }
        
        await websocket.send_json(message)
        
    except Exception as e:
        logger.error(f"[{call_id}] Failed to send audio to FreeSWITCH: {e}", exc_info=True)


async def handle_call_loop(
    websocket: WebSocket,
    call_id: str,
):
    """
    Основной цикл обработки звонка.
    
    Args:
        websocket: WebSocket соединение с FreeSWITCH
        call_id: ID звонка
    """
    session = active_calls.get(call_id)
    if not session:
        logger.error(f"[{call_id}] Session not found in handle_call_loop")
        return
    
    # Сохраняем websocket в сессии
    session.freeswitch_ws = websocket
    
    logger.info(f"[{call_id}] Starting call loop")
    
    # Подключаемся к ASR Gateway
    asr_ws = await connect_to_asr(call_id)
    if not asr_ws:
        logger.error(f"[{call_id}] Failed to connect to ASR, closing call")
        return
    
    session.asr_ws = asr_ws
    
    # Запускаем задачу для получения транскриптов от ASR
    transcript_task = asyncio.create_task(
        receive_transcripts_loop(asr_ws, call_id)
    )
    
    try:
        while True:
            # Получаем сообщение от FreeSWITCH
            try:
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                # Проверяем транскрипты
                continue
            except WebSocketDisconnect:
                logger.info(f"[{call_id}] FreeSWITCH disconnected")
                break
            
            # Обрабатываем входящий аудио
            if message.get("type") == "audio" and message.get("direction") == "input":
                audio_data = message.get("data")
                if audio_data:
                    # Декодируем аудио
                    audio = decode_audio_data(audio_data, config.audio.sample_rate)
                    
                    # Проверяем barge-in (если агент говорит)
                    if session.is_speaking and config.barge_in.enabled and barge_in_vad:
                        # Используем VAD для детектирования речи клиента
                        if barge_in_vad.detect_speech(audio):
                            logger.info(
                                f"[{call_id}] Barge-in detected! Stopping TTS playback"
                            )
                            
                            # Останавливаем текущий TTS task
                            if session.current_tts_task and not session.current_tts_task.done():
                                session.current_tts_task.cancel()
                                try:
                                    await session.current_tts_task
                                except asyncio.CancelledError:
                                    pass
                            
                            # Сбрасываем состояние
                            session.is_speaking = False
                            session.current_tts_task = None
                            
                            # Сбрасываем VAD detector для нового цикла
                            barge_in_vad.reset()
                            
                            # Логируем событие для мониторинга
                            logger.info(
                                f"[{call_id}] TTS stopped, switching to listening mode",
                                extra={
                                    "context": {
                                        "event": "barge_in",
                                        "call_id": call_id,
                                    }
                                },
                            )
                    
                    # Отправляем в ASR Gateway
                    await send_audio_to_asr(asr_ws, audio, call_id)
            
            # Обрабатываем другие типы сообщений
            elif message.get("type") == "event":
                event_type = message.get("event")
                logger.debug(f"[{call_id}] FreeSWITCH event: {event_type}")
                
                if event_type == "CHANNEL_HANGUP":
                    logger.info(f"[{call_id}] Call ended")
                    break
    
    except WebSocketDisconnect:
        logger.info(f"[{call_id}] WebSocket disconnected")
    except Exception as e:
        logger.error(f"[{call_id}] Call loop error: {e}", exc_info=True)
    finally:
        # Очистка
        transcript_task.cancel()
        try:
            await transcript_task
        except asyncio.CancelledError:
            pass
        
        if asr_ws:
            await asr_ws.close()
        
        # Удаляем сессию
        if call_id in active_calls:
            del active_calls[call_id]
        
        logger.info(f"[{call_id}] Call loop ended")


async def receive_transcripts_loop(
    asr_ws: websockets.WebSocketClientProtocol,
    call_id: str,
):
    """
    Цикл получения транскриптов от ASR Gateway.
    
    Args:
        asr_ws: WebSocket соединение с ASR Gateway
        call_id: ID звонка
    """
    session = active_calls.get(call_id)
    if not session:
        return
    
    partial_transcript_buffer = ""
    last_final_time = 0.0
    last_transcript_time = 0.0
    transcript_history: list[tuple[float, str]] = []  # (timestamp, transcript) для расчета скорости речи
    
    try:
        while True:
            # Получаем транскрипт
            transcript = await receive_transcript_from_asr(asr_ws, call_id)
            
            current_time = asyncio.get_event_loop().time()
            
            if transcript:
                partial_transcript_buffer = transcript
                last_final_time = current_time
                
                # Отслеживаем историю транскриптов для адаптации порога
                if config.endpointing.adaptive_enabled:
                    transcript_history.append((current_time, transcript))
                    # Храним только последние 5 транскриптов для расчета скорости
                    if len(transcript_history) > 5:
                        transcript_history.pop(0)
                    
                    last_transcript_time = current_time
                
                logger.debug(f"[{call_id}] Partial transcript: {transcript[:50]}...")
            
            # Вычисляем адаптивный порог тишины
            silence_threshold = config.endpointing.base_silence_threshold_ms / 1000.0
            
            if config.endpointing.adaptive_enabled and transcript_history:
                # Рассчитываем скорость речи на основе последних транскриптов
                if len(transcript_history) >= 2:
                    time_span = transcript_history[-1][0] - transcript_history[0][0]
                    if time_span > 0:
                        # Средняя скорость обновления транскриптов (обновлений в секунду)
                        update_rate = len(transcript_history) / time_span
                        
                        # Адаптируем порог: быстрая речь = меньший порог
                        if update_rate > 5.0:  # Быстрая речь (> 5 обновлений/сек)
                            silence_threshold = config.endpointing.fast_speech_threshold_ms / 1000.0
                        elif update_rate < 2.0:  # Медленная речь (< 2 обновлений/сек)
                            silence_threshold = config.endpointing.slow_speech_threshold_ms / 1000.0
                        else:
                            silence_threshold = config.endpointing.base_silence_threshold_ms / 1000.0
                
                # Ограничиваем порог минимальным и максимальным значениями
                silence_threshold = max(
                    config.endpointing.min_threshold_ms / 1000.0,
                    min(silence_threshold, config.endpointing.max_threshold_ms / 1000.0)
                )
            
            # Если прошло достаточно времени без новых транскриптов, считаем финальным
            if (
                partial_transcript_buffer
                and (current_time - last_final_time) > silence_threshold
            ):
                final_transcript = partial_transcript_buffer
                partial_transcript_buffer = ""
                transcript_history.clear()  # Очищаем историю после финализации
                
                logger.info(
                    f"[{call_id}] Final transcript (threshold={silence_threshold*1000:.0f}ms): {final_transcript}"
                )
                
                # Обрабатываем диалог
                agent_message = await process_dialog(final_transcript, call_id)
                
                if agent_message:
                    # Синтезируем речь
                    audio = await synthesize_speech(agent_message, call_id)
                    
                    if audio is not None:
                        # Создаем задачу для отправки TTS (можно прервать при barge-in)
                        async def send_tts_audio():
                            """Отправляет TTS аудио по чанкам."""
                            task = asyncio.current_task()
                            try:
                                session.is_speaking = True
                                
                                # Отправляем аудио в FreeSWITCH по чанкам
                                chunk_size = int(
                                    config.audio.sample_rate * config.audio.chunk_size_ms / 1000
                                )
                                
                                for i in range(0, len(audio), chunk_size):
                                    # Проверяем, не был ли task отменен
                                    if task and task.cancelled():
                                        logger.debug(f"[{call_id}] TTS task cancelled, stopping playback")
                                        break
                                    
                                    chunk = audio[i : i + chunk_size]
                                    
                                    # Отправляем чанк в FreeSWITCH
                                    await send_audio_to_freeswitch(
                                        session.freeswitch_ws,
                                        chunk,
                                        call_id,
                                    )
                                    
                                    await asyncio.sleep(config.audio.chunk_size_ms / 1000.0)
                                
                                session.is_speaking = False
                                logger.debug(f"[{call_id}] TTS playback completed")
                                
                            except asyncio.CancelledError:
                                logger.debug(f"[{call_id}] TTS playback cancelled")
                                session.is_speaking = False
                                raise
                            except Exception as e:
                                logger.error(f"[{call_id}] Error during TTS playback: {e}", exc_info=True)
                                session.is_speaking = False
                        
                        # Создаем и сохраняем task
                        session.current_tts_task = asyncio.create_task(send_tts_audio())
                        
                        # Ждем завершения или отмены task
                        try:
                            await session.current_tts_task
                        except asyncio.CancelledError:
                            logger.debug(f"[{call_id}] TTS task was cancelled")
                        finally:
                            session.current_tts_task = None
            
            await asyncio.sleep(0.01)  # Небольшая задержка
    
    except asyncio.CancelledError:
        logger.info(f"[{call_id}] Transcript loop cancelled")
    except Exception as e:
        logger.error(f"[{call_id}] Transcript loop error: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    global config, telemetry, http_client, barge_in_vad
    
    logger.info("Starting FreeSWITCH Bridge...")
    
    try:
        # Загружаем конфигурацию
        config_path = Path(__file__).parent / "config.yaml"
        config = load_and_validate_config(config_path, FreeSWITCHConfig, "FREESWITCH_BRIDGE")
        
        logger.info("Configuration loaded")
        
        # Инициализация telemetry
        telemetry = TelemetryManager("freeswitch_bridge")
        
        # Инициализация HTTP клиента
        http_client = httpx.AsyncClient(timeout=30.0)
        
        # Инициализация VAD detector для barge-in detection
        if config.barge_in.enabled:
            try:
                barge_in_vad = VADDetector(
                    threshold=config.barge_in.vad_threshold,
                    min_speech_duration_ms=config.barge_in.min_speech_duration_ms,
                    sample_rate=config.audio.sample_rate,
                    device=config.barge_in.device,
                    use_onnx=config.barge_in.use_onnx,
                    logger=logger,
                )
                logger.info("Barge-in VAD detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize barge-in VAD detector: {e}", exc_info=True)
                logger.warning("Barge-in detection will be disabled")
                barge_in_vad = None
        else:
            logger.info("Barge-in detection disabled")
        
        logger.info("FreeSWITCH Bridge started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start FreeSWITCH Bridge: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down FreeSWITCH Bridge...")
        if http_client:
            await http_client.aclose()
        if barge_in_vad:
            barge_in_vad.reset()


# FastAPI app
app = FastAPI(
    title="FreeSWITCH Bridge",
    description="WebSocket bridge for FreeSWITCH mod_audio_fork integration",
    version="0.1.0",
    lifespan=lifespan,
)

health_checker = HealthChecker("freeswitch_bridge", "0.1.0")
app.include_router(health_checker.create_router())


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    metrics_data, content_type = telemetry.get_prometheus_metrics()
    return Response(content=metrics_data, media_type=content_type)


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """
    WebSocket endpoint для FreeSWITCH mod_audio_fork.
    
    Обрабатывает двунаправленный аудио поток:
    - Входящий: FreeSWITCH → ASR Gateway → Policy Engine
    - Исходящий: TTS Gateway → FreeSWITCH
    """
    await websocket.accept()
    
    # Генерируем ID звонка
    call_id = str(uuid.uuid4())
    
    logger.info(f"[{call_id}] FreeSWITCH connected")
    
    # Создаем сессию с websocket
    session = CallSession(call_id, websocket)
    active_calls[call_id] = session
    
    try:
        # Запускаем основной цикл обработки звонка
        await handle_call_loop(websocket, call_id)
    
    except WebSocketDisconnect:
        logger.info(f"[{call_id}] FreeSWITCH disconnected")
    except Exception as e:
        logger.error(f"[{call_id}] WebSocket error: {e}", exc_info=True)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        # Очистка
        if call_id in active_calls:
            del active_calls[call_id]
        logger.info(f"[{call_id}] WebSocket closed")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "freeswitch_bridge",
        "status": "running",
        "active_calls": len(active_calls),
    }


@app.get("/calls")
async def list_calls():
    """Список активных звонков."""
    return {
        "active_calls": [
            {
                "call_id": call_id,
                "session_id": session.session_id,
                "is_speaking": session.is_speaking,
            }
            for call_id, session in active_calls.items()
        ]
    }


def main():
    """Точка входа."""
    uvloop.install()
    
    import uvicorn
    
    host = os.getenv("FREESWITCH_BRIDGE_HOST", "0.0.0.0")
    port = int(os.getenv("FREESWITCH_BRIDGE_PORT", 8004))
    
    uvicorn.run(
        "src.freeswitch_bridge.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    import os
    main()

