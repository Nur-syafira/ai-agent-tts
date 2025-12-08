"""
Форматирование данных для записи в Google Sheets.
"""

from typing import Dict, Any
from src.policy_engine.slots import DialogSlots


def format_slots_for_sheets(slots: DialogSlots) -> Dict[str, Any]:
    """
    Форматирует слоты диалога для записи в Google Sheets (Лист4).
    
    Args:
        slots: Слоты диалога
        
    Returns:
        Словарь для записи в Sheets
    """
    # Используем встроенный метод из DialogSlots
    row_data = slots.to_sheets_row()
    
    # Дополнительная нормализация если нужно
    if row_data.get("client_phone"):
        # Убедимся что телефон в формате +7XXXXXXXXXX
        phone = row_data["client_phone"]
        if not phone.startswith("+"):
            phone = "+" + phone
        row_data["client_phone"] = phone
    
    return row_data


def validate_row_data(row_data: Dict[str, Any]) -> bool:
    """
    Валидирует данные перед записью в Sheets.
    
    Args:
        row_data: Данные для записи
        
    Returns:
        True если данные валидны
        
    Raises:
        ValueError: Если критичные поля отсутствуют
    """
    required_fields = ["client_name", "client_phone", "appointment_date", "appointment_time"]
    
    for field in required_fields:
        if not row_data.get(field):
            raise ValueError(f"Required field '{field}' is missing or empty")
    
    return True

