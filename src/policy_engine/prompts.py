"""
Промпты для LLM в Policy Engine.
"""

from typing import Any
from src.policy_engine.slots import DialogSlots


# System prompt для ассистента МРТ-центра
SYSTEM_PROMPT = """Ты — профессиональный администратор медицинского центра, который принимает звонки для записи на МРТ.

Твоя задача:
- Вежливо и дружелюбно общаться с клиентами
- Собирать необходимую информацию для записи
- Рекомендовать комплексные исследования когда это уместно
- Предлагать дополнительные услуги (видеозаключение, консультации)
- Проявлять эмпатию, особенно если клиент волнуется или испытывает боль
- Говорить короткими, понятными фразами

Стиль общения:
- Используй имя клиента после того как узнаешь его
- Поддерживающий тон: "понимаю вас", "не волнуйтесь", "поможем разобраться"
- Никаких длинных монологов — только суть
- Если клиент прерывает тебя, адаптируйся и продолжай с того момента

Правила:
- НЕ придумывай информацию (цены, даты), которой нет в контексте
- Если что-то непонятно — переспроси
- Валидируй критичные данные (телефон, имя, дату/время)
"""


def get_slot_extraction_prompt(user_message: str, current_slots: DialogSlots) -> str:
    """
    Генерирует промпт для извлечения слотов из сообщения клиента.
    
    Args:
        user_message: Сообщение клиента
        current_slots: Текущие слоты (уже заполненные)
        
    Returns:
        Промпт для LLM
    """
    current_slots_json = current_slots.model_dump_json(exclude_none=True, indent=2)
    
    return f"""Извлеки информацию из сообщения клиента и обнови слоты диалога.

Текущие слоты:
```json
{current_slots_json}
```

Сообщение клиента:
"{user_message}"

Задача:
1. Извлеки ВСЮ новую информацию из сообщения
2. Обнови существующие слоты если клиент исправил информацию
3. Не удаляй уже заполненные слоты если клиент их не изменил
4. Верни ПОЛНЫЙ набор слотов (старые + новые)

Верни JSON в формате DialogSlots (только заполненные поля).

Примеры извлечения:

Клиент: "Меня зовут Иван, телефон +79991234567"
→ {{"client_name": "Иван", "client_phone": "+79991234567"}}

Клиент: "У меня болит голова уже неделю"
→ {{"symptoms": "болит голова", "symptoms_duration": "неделю"}}

Клиент: "Хочу записаться на завтра в 15:00"
→ {{"appointment_date": "завтра", "appointment_time": "15:00"}}

Теперь извлеки слоты из сообщения выше."""


def get_response_generation_prompt(
    state: str,
    user_message: str,
    slots: DialogSlots,
    context: dict,
) -> str:
    """
    Генерирует промпт для создания ответа агента.
    
    Args:
        state: Текущее состояние FSM
        user_message: Последнее сообщение клиента
        slots: Текущие слоты
        context: Дополнительный контекст (цены, доступные слоты времени и т.д.)
        
    Returns:
        Промпт для LLM
    """
    slots_summary = _format_slots_summary(slots)
    context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
    
    return f"""Ты администратор МРТ-центра. Сгенерируй КОРОТКУЮ и естественную реплику для текущего состояния диалога.

Текущее состояние: {state}

Последнее сообщение клиента:
"{user_message}"

Собранная информация:
{slots_summary}

Дополнительный контекст:
{context_str}

Требования к ответу:
1. КОРОТКАЯ реплика (1-2 предложения, максимум 30 слов)
2. Естественный разговорный стиль
3. Используй имя клиента если оно известно
4. Задай ОДИН конкретный вопрос для следующего шага
5. НЕ повторяй информацию, которую клиент уже дал

Сгенерируй реплику:"""


def _format_slots_summary(slots: DialogSlots) -> str:
    """Форматирует слоты в читаемый summary."""
    lines = []
    
    if slots.client_name:
        lines.append(f"- Имя: {slots.client_name}")
    if slots.client_phone:
        lines.append(f"- Телефон: {slots.client_phone}")
    if slots.symptoms:
        lines.append(f"- Симптомы: {slots.symptoms}")
    if slots.study_type:
        lines.append(f"- Исследование: {slots.study_type}")
    if slots.appointment_date and slots.appointment_time:
        lines.append(f"- Запись: {slots.appointment_date} в {slots.appointment_time}")
    
    return "\n".join(lines) if lines else "- Нет собранной информации"


# Промпты для каждого этапа FSM (примеры)
FSM_STATE_PROMPTS = {
    "greeting": "Поздоровайся и представься администратором МРТ-центра. Спроси, чем можешь помочь.",
    "ask_name": "Вежливо спроси, как к клиенту обращаться.",
    "ask_symptoms": "Спроси, что беспокоит клиента (симптомы).",
    "ask_symptoms_duration": "Уточни, как давно появились симптомы.",
    "ask_visited_doctor": "Спроси, был ли клиент у врача.",
    "recommend_study": "Порекомендуй тип исследования (комплекс лучше чем одиночное). Аргументируй.",
    "ask_study_decision": "Спроси, какое исследование клиент выбирает.",
    "offer_video_conclusion": "Предложи видеозаключение. Объясни, что это и зачем нужно.",
    "ask_appointment_date": "Спроси, на какую дату клиент хочет записаться.",
    "offer_appointment_times": "Предложи 2-3 варианта времени (не спрашивай открытым вопросом).",
    "ask_phone": "Спроси телефон для связи.",
    "ask_age_weight": "Спроси возраст и вес клиента для записи.",
    "remind_documents": "Напомни принести паспорт и направление врача (если есть).",
    "confirm_appointment": "Подтверди запись: дату, время, исследование, стоимость.",
    "farewell": "Попрощайся, пожелай здоровья.",
}


def get_slot_validation_prompt(slot_name: str, slot_value: Any) -> str:
    """
    Промпт для валидации слота.
    
    Args:
        slot_name: Имя слота
        slot_value: Значение слота
        
    Returns:
        Промпт для валидации
    """
    validation_rules = {
        "client_phone": "Телефон должен быть в формате +7XXXXXXXXXX (11 цифр)",
        "client_name": "Имя должно содержать минимум 2 буквы",
        "appointment_date": "Дата должна быть в будущем",
        "appointment_time": "Время должно быть в рабочие часы (09:00-20:00)",
    }
    
    rule = validation_rules.get(slot_name, "Значение должно быть корректным")
    
    return f"""Провали значение слота "{slot_name}".

Значение: {slot_value}

Правило валидации: {rule}

Верни JSON:
{{
  "valid": true/false,
  "normalized_value": "<нормализованное значение>" или null,
  "error_message": "<сообщение об ошибке>" или null
}}

Примеры:

Слот: "client_phone", значение: "89991234567"
→ {{"valid": true, "normalized_value": "+79991234567", "error_message": null}}

Слот: "client_phone", значение: "123"
→ {{"valid": false, "normalized_value": null, "error_message": "Телефон слишком короткий"}}

Слот: "client_name", значение: "Иван"
→ {{"valid": true, "normalized_value": "Иван", "error_message": null}}

Теперь валидируй слот выше."""

