"""
Пример кастомизации промптов для своего домена.

Этот пример показывает, как настроить промпты под свою задачу.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy_engine.prompts import (
    SYSTEM_PROMPT,
    get_slot_extraction_prompt,
    get_response_generation_prompt,
)


# Кастомный system prompt для вашего домена
CUSTOM_SYSTEM_PROMPT = """Ты — профессиональный помощник для [ваш домен].

Твоя задача:
- [Ваша задача 1]
- [Ваша задача 2]
- [Ваша задача 3]

Стиль общения:
- [Ваш стиль 1]
- [Ваш стиль 2]

Правила:
- [Ваше правило 1]
- [Ваше правило 2]
"""


def custom_slot_extraction_prompt(user_message: str, current_slots: dict) -> str:
    """Кастомный промпт для извлечения слотов."""
    return f"""Извлеки информацию из сообщения клиента.

Текущие данные:
{current_slots}

Сообщение клиента:
"{user_message}"

Верни JSON с извлеченными данными в формате:
{{
  "field1": "значение1",
  "field2": "значение2"
}}
"""


def custom_response_prompt(state: str, user_message: str, slots: dict) -> str:
    """Кастомный промпт для генерации ответа."""
    return f"""Сгенерируй ответ для состояния: {state}

Сообщение клиента: "{user_message}"
Собранные данные: {slots}

Требования:
- Короткий ответ (1-2 предложения)
- Естественный стиль
- Задай один вопрос для следующего шага
"""


# Пример использования
if __name__ == "__main__":
    # Использование стандартных промптов
    print("=== Стандартный промпт ===")
    print(SYSTEM_PROMPT[:200] + "...")
    
    # Использование кастомных промптов
    print("\n=== Кастомный промпт ===")
    print(CUSTOM_SYSTEM_PROMPT[:200] + "...")
    
    # Пример извлечения слотов
    print("\n=== Извлечение слотов ===")
    prompt = custom_slot_extraction_prompt(
        "Меня зовут Иван, телефон +79991234567",
        {}
    )
    print(prompt[:300] + "...")

