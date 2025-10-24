"""
Policy Engine - FastAPI сервер для управления диалогом.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvloop
from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import redis.asyncio as aioredis
import json
import os
from dotenv import load_dotenv

from src.shared.logging_config import setup_logging
from src.shared.config_loader import load_and_validate_config
from src.shared.health import HealthChecker
from src.shared.metrics import TelemetryManager
from src.policy_engine.fsm import DialogFSM, DialogState
from src.policy_engine.slots import DialogSlots
from src.policy_engine.prompts import (
    SYSTEM_PROMPT,
    get_slot_extraction_prompt,
    get_response_generation_prompt,
)
from src.llm_service.main import LLMClient

load_dotenv()

logger = setup_logging("policy_engine")


class PolicyConfig(BaseModel):
    """Pydantic модель для конфигурации Policy Engine."""
    
    class LLMConfig(BaseModel):
        base_url: str
        model_name: str
        temperature: float
        max_tokens: int
        structured_temperature: float
        structured_max_tokens: int
    
    class ServicesConfig(BaseModel):
        asr_url: str
        tts_url: str
    
    class FSMConfig(BaseModel):
        user_response_timeout: int
        max_retries: int
        validate_slots: bool
    
    class NotifierConfig(BaseModel):
        enabled: bool
        async_write: bool
    
    class ServerConfig(BaseModel):
        host: str
        port: int
    
    class StateStorageConfig(BaseModel):
        redis_url: str
        session_ttl: int
    
    llm: LLMConfig
    services: ServicesConfig
    fsm: FSMConfig
    notifier: NotifierConfig
    server: ServerConfig
    state_storage: StateStorageConfig


# Глобальные переменные
config: PolicyConfig
llm_client: LLMClient
redis_client: aioredis.Redis
telemetry: TelemetryManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    global config, llm_client, redis_client, telemetry
    
    logger.info("Starting Policy Engine...")
    
    try:
        # Загружаем конфигурацию
        config_path = Path(__file__).parent / "config.yaml"
        config = load_and_validate_config(config_path, PolicyConfig, "POLICY_ENGINE")
        
        logger.info("Configuration loaded")
        
        # Инициализация telemetry
        telemetry = TelemetryManager("policy_engine")
        
        # Инициализация LLM клиента
        llm_client = LLMClient(
            base_url=config.llm.base_url,
            api_key=None,
        )
        
        # Проверяем доступность LLM
        is_llm_ready = await llm_client.health_check()
        if not is_llm_ready:
            logger.warning("LLM service is not available yet")
        else:
            logger.info("LLM service is ready")
        
        # Инициализация Redis для хранения состояний сессий
        redis_client = await aioredis.from_url(
            config.state_storage.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await redis_client.ping()
        logger.info("Redis connected")
        
        logger.info("Policy Engine started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Policy Engine: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down Policy Engine...")
        if redis_client:
            await redis_client.close()


# FastAPI app
app = FastAPI(
    title="Policy Engine",
    description="Dialog orchestrator with FSM for MRI appointment",
    version="0.1.0",
    lifespan=lifespan,
)

health_checker = HealthChecker("policy_engine", "0.1.0")
app.include_router(health_checker.create_router())


class SessionState(BaseModel):
    """Состояние сессии диалога."""
    
    session_id: str
    fsm_state: DialogState
    slots: DialogSlots
    conversation_history: list[dict] = []


class DialogRequest(BaseModel):
    """Запрос на обработку сообщения в диалоге."""
    
    session_id: str
    user_message: str


class DialogResponse(BaseModel):
    """Ответ от Policy Engine."""
    
    session_id: str
    agent_message: str
    current_state: DialogState
    slots: DialogSlots
    is_complete: bool


async def get_session_state(session_id: str) -> SessionState:
    """
    Получает состояние сессии из Redis.
    
    Args:
        session_id: ID сессии
        
    Returns:
        Состояние сессии
    """
    key = f"session:{session_id}"
    data = await redis_client.get(key)
    
    if data:
        state_dict = json.loads(data)
        return SessionState(**state_dict)
    else:
        # Новая сессия
        return SessionState(
            session_id=session_id,
            fsm_state=DialogState.START,
            slots=DialogSlots(),
            conversation_history=[],
        )


async def save_session_state(state: SessionState):
    """
    Сохраняет состояние сессии в Redis.
    
    Args:
        state: Состояние сессии
    """
    key = f"session:{state.session_id}"
    data = state.model_dump_json()
    
    await redis_client.setex(
        key,
        config.state_storage.session_ttl,
        data,
    )


async def extract_slots(user_message: str, current_slots: DialogSlots) -> DialogSlots:
    """
    Извлекает слоты из сообщения пользователя с помощью LLM.
    
    Args:
        user_message: Сообщение пользователя
        current_slots: Текущие слоты
        
    Returns:
        Обновлённые слоты
    """
    prompt = get_slot_extraction_prompt(user_message, current_slots)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    try:
        response = await llm_client.generate_structured(
            messages=messages,
            temperature=config.llm.structured_temperature,
            max_tokens=config.llm.structured_max_tokens,
        )
        
        # Парсим JSON ответ
        slots_dict = json.loads(response)
        
        # Объединяем со старыми слотами
        updated_dict = {**current_slots.model_dump(exclude_none=True), **slots_dict}
        
        return DialogSlots(**updated_dict)
        
    except Exception as e:
        logger.error(f"Slot extraction error: {e}", exc_info=True)
        return current_slots


async def generate_agent_response(
    fsm: DialogFSM,
    user_message: str,
    slots: DialogSlots,
) -> str:
    """
    Генерирует ответ агента с помощью LLM.
    
    Args:
        fsm: FSM объект
        user_message: Сообщение пользователя
        slots: Текущие слоты
        
    Returns:
        Ответ агента
    """
    context = fsm.get_state_context()
    prompt = get_response_generation_prompt(
        state=fsm.current_state.value,
        user_message=user_message,
        slots=slots,
        context=context,
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    try:
        response = await llm_client.generate(
            messages=messages,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Response generation error: {e}", exc_info=True)
        return "Извините, произошла ошибка. Можете повторить?"


@app.post("/dialog", response_model=DialogResponse)
async def process_dialog(request: DialogRequest):
    """
    Обрабатывает сообщение в диалоге.
    
    Args:
        request: Запрос с сообщением пользователя
        
    Returns:
        Ответ агента
    """
    try:
        # Получаем состояние сессии
        session_state = await get_session_state(request.session_id)
        
        # Инициализируем FSM
        fsm = DialogFSM(logger=logger)
        fsm.current_state = session_state.fsm_state
        
        # Извлекаем слоты из сообщения
        with telemetry.trace_span("extract_slots"):
            updated_slots = await extract_slots(request.user_message, session_state.slots)
        
        # Определяем следующее состояние
        next_state = fsm.get_next_state(updated_slots, request.user_message)
        
        if next_state:
            fsm.transition(next_state)
        
        # Генерируем ответ агента
        with telemetry.trace_span("generate_response"):
            agent_message = await generate_agent_response(
                fsm=fsm,
                user_message=request.user_message,
                slots=updated_slots,
            )
        
        # Обновляем историю
        session_state.conversation_history.append(
            {"role": "user", "content": request.user_message}
        )
        session_state.conversation_history.append(
            {"role": "assistant", "content": agent_message}
        )
        
        # Сохраняем состояние
        session_state.fsm_state = fsm.current_state
        session_state.slots = updated_slots
        await save_session_state(session_state)
        
        # Если диалог завершён и слоты заполнены, отправляем в notifier
        is_complete = updated_slots.is_complete() and fsm.is_terminal_state()
        
        if is_complete and config.notifier.enabled:
            # TODO: Интеграция с notifier для записи в Google Sheets
            logger.info(f"Dialog complete for session {request.session_id}")
        
        return DialogResponse(
            session_id=request.session_id,
            agent_message=agent_message,
            current_state=fsm.current_state,
            slots=updated_slots,
            is_complete=is_complete,
        )
        
    except Exception as e:
        logger.error(f"Dialog processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Получает текущее состояние сессии."""
    try:
        state = await get_session_state(session_id)
        return state.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {e}",
        )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Удаляет сессию."""
    key = f"session:{session_id}"
    await redis_client.delete(key)
    return {"status": "deleted", "session_id": session_id}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "policy_engine", "status": "running"}


def main():
    """Точка входа."""
    uvloop.install()
    
    import uvicorn
    
    host = os.getenv("POLICY_ENGINE_HOST", "0.0.0.0")
    port = int(os.getenv("POLICY_ENGINE_PORT", 8003))
    
    uvicorn.run(
        "src.policy_engine.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()

