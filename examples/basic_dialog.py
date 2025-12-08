"""
Базовый пример использования Policy Engine для обработки диалога.

Этот пример показывает минимальный код для запуска диалога с агентом.
"""

import asyncio
import sys
from pathlib import Path

# Добавить корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy_engine.main import PolicyEngine
from src.shared.config_loader import load_and_validate_config


async def main():
    """Пример базового диалога."""
    
    # Инициализация Policy Engine
    # Автоматически загружает конфигурацию из config.yaml
    engine = PolicyEngine()
    
    session_id = "example-session-123"
    
    # Симуляция диалога
    messages = [
        "Здравствуйте, хочу записаться на МРТ",
        "Меня зовут Иван Петров",
        "У меня болит голова",
        "Уже неделю",
        "Да, согласен на комплексное исследование",
        "На завтра в 15:00",
        "+79991234567",
    ]
    
    print("🤖 Начинаем диалог с агентом...\n")
    
    for i, user_message in enumerate(messages, 1):
        print(f"👤 Пользователь ({i}): {user_message}")
        
        # Обработка сообщения
        response = await engine.process_message(
            session_id=session_id,
            user_message=user_message
        )
        
        print(f"🤖 Агент: {response.agent_message}\n")
        
        # Показать заполненные слоты
        if response.slots.model_dump(exclude_none=True):
            print(f"📋 Слоты: {response.slots.model_dump(exclude_none=True)}\n")
        
        # Небольшая задержка для реалистичности
        await asyncio.sleep(0.5)
    
    print("✅ Диалог завершён!")


if __name__ == "__main__":
    asyncio.run(main())

