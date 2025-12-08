"""
Policy Engine - FastAPI сервер для управления диалогом.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvloop
import asyncio
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from contextlib import asynccontextmanager
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Any
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
    get_system_prompt,
    get_agent_role,
    get_slot_extraction_prompt,
    get_response_generation_prompt,
)
from src.llm_service.main import LLMClient

load_dotenv()

logger = setup_logging("policy_engine")


class PolicyConfig(BaseSettings):
    """Pydantic Settings модель для конфигурации Policy Engine."""
    
    class LLMConfig(BaseModel):
        base_url: str
        model_name: str
        temperature: float
        max_tokens: int
        structured_temperature: float
        structured_max_tokens: int
        max_history_turns: int = 10  # Количество последних реплик для передачи в LLM
    
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
        credentials_path: str
        spreadsheet_id: str
        worksheet_name: str
    
    class ServerConfig(BaseModel):
        host: str
        port: int
    
    class StateStorageConfig(BaseModel):
        redis_url: str
        session_ttl: int
    
    class PromptsConfig(BaseModel):
        system_prompt: Optional[str] = None
        agent_role: Optional[str] = None
    
    llm: LLMConfig
    services: ServicesConfig
    fsm: FSMConfig
    notifier: NotifierConfig
    server: ServerConfig
    state_storage: StateStorageConfig
    prompts: Optional[PromptsConfig] = None


# Глобальные переменные для сервисов
config: Optional[PolicyConfig] = None
llm_client: Optional[LLMClient] = None
redis_client: Optional[aioredis.Redis] = None
telemetry: Optional[TelemetryManager] = None
sheets_client: Optional[Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    global config, llm_client, redis_client, telemetry, sheets_client
    
    logger.info("Starting Policy Engine...")
    
    redis_client = None  # Инициализируем как None
    
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
            model_name=config.llm.model_name,  # Передаем имя модели из конфига
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
        
        # Инициализация Google Sheets клиента (если notifier включен)
        if config.notifier.enabled:
            try:
                from src.notifier.sheets_client import GoogleSheetsClient
                
                sheets_client = GoogleSheetsClient(
                    credentials_path=config.notifier.credentials_path,
                    spreadsheet_id=config.notifier.spreadsheet_id,
                    worksheet_name=config.notifier.worksheet_name,
                    logger=logger,
                )
                logger.info("Google Sheets client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Sheets client: {e}. Notifier will be disabled.")
                config.notifier.enabled = False
        
        logger.info("Policy Engine started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Policy Engine: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down Policy Engine...")
        try:
            if redis_client is not None:
                await redis_client.close()
        except:
            pass


# FastAPI app
app = FastAPI(
    title="Policy Engine",
    description="Dialog orchestrator with FSM for MRI appointment",
    version="0.1.0",
    lifespan=lifespan,
)

health_checker = HealthChecker("policy_engine", "0.1.0")
app.include_router(health_checker.create_router())


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    metrics_data, content_type = telemetry.get_prometheus_metrics()
    return Response(content=metrics_data, media_type=content_type)


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


async def extract_slots(
    user_message: str,
    current_slots: DialogSlots,
    conversation_history: Optional[list[dict]] = None,
) -> DialogSlots:
    """
    Извлекает слоты из сообщения пользователя с помощью LLM.
    
    Args:
        user_message: Сообщение пользователя
        current_slots: Текущие слоты
        conversation_history: История диалога для контекста (опционально)
        
    Returns:
        Обновлённые слоты
    """
    prompt = get_slot_extraction_prompt(user_message, current_slots)
    
    # Получаем system prompt из конфига или используем дефолтный
    system_prompt = get_system_prompt(
        config.prompts.system_prompt if config.prompts else None
    )
    
    # Формируем messages с историей диалога для контекста
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    # Добавляем историю диалога если она есть
    if conversation_history:
        # Берем последние N реплик из истории
        max_history = config.llm.max_history_turns
        recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
        messages.extend(recent_history)
    
    # Добавляем текущий промпт
    # Для Qwen3-16B-A3B-abliterated-AWQ добавляем "no think" режим для ускорения
    # Это отключает reasoning и ускоряет генерацию (согласно документации модели)
    prompt_with_no_think = prompt + "\n<think>\n\n</think>\n"
    messages.append({"role": "user", "content": prompt_with_no_think})
    
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
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Генерирует ответ агента с помощью LLM.
    
    Args:
        fsm: FSM объект
        user_message: Сообщение пользователя
        slots: Текущие слоты
        conversation_history: История диалога для контекста (опционально)
        
    Returns:
        Ответ агента
    """
    # Получаем system prompt и роль агента из конфига
    system_prompt = get_system_prompt(
        config.prompts.system_prompt if config.prompts else None
    )
    agent_role = get_agent_role(
        config.prompts.agent_role if config.prompts else None
    )
    
    context = fsm.get_state_context()
    prompt = get_response_generation_prompt(
        state=fsm.current_state.value,
        user_message=user_message,
        slots=slots,
        context=context,
        agent_role=agent_role,
    )
    
    # Формируем messages с историей диалога
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    # Добавляем историю диалога если она есть
    if conversation_history:
        # Берем последние N реплик из истории
        max_history = config.llm.max_history_turns
        recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
        messages.extend(recent_history)
    
    # Добавляем текущий промпт
    # Для Qwen3-16B-A3B-abliterated-AWQ добавляем "no think" режим для ускорения
    # Это отключает reasoning и ускоряет генерацию (согласно документации модели)
    prompt_with_no_think = prompt + "\n<think>\n\n</think>\n"
    messages.append({"role": "user", "content": prompt_with_no_think})
    
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
            updated_slots = await extract_slots(
                request.user_message,
                session_state.slots,
                conversation_history=session_state.conversation_history,
            )
        
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
                conversation_history=session_state.conversation_history,
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
        
        if is_complete and config.notifier.enabled and sheets_client:
            try:
                from src.notifier.formatter import format_slots_for_sheets, validate_row_data
                
                # Форматируем слоты для Google Sheets
                row_data = format_slots_for_sheets(updated_slots)
                
                # Валидируем данные
                try:
                    validate_row_data(row_data)
                except ValueError as e:
                    logger.warning(f"Row data validation failed: {e}. Skipping Sheets write.")
                else:
                    # Асинхронная запись (не блокирует диалог)
                    if config.notifier.async_write:
                        asyncio.create_task(
                            sheets_client.append_row(row_data)
                        )
                        logger.info(f"Queued Sheets write for session {request.session_id}")
                    else:
                        await sheets_client.append_row(row_data)
                        logger.info(f"Wrote to Sheets for session {request.session_id}")
                        
            except Exception as e:
                logger.error(f"Failed to write to Google Sheets: {e}", exc_info=True)
                # Не прерываем диалог из-за ошибки записи в Sheets
        
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

