"""
Pydantic модели для слотов диалога (по script_evaluation_type_A.md).
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date, time
import re


class DialogSlots(BaseModel):
    """
    Слоты для диалога записи на МРТ.
    
    Основано на script_evaluation_type_A.md (30 этапов).
    """
    
    # Основные сущности
    client_name: Optional[str] = Field(None, description="Имя клиента")
    client_phone: Optional[str] = Field(None, description="Телефон клиента")
    client_age: Optional[int] = Field(None, description="Возраст клиента", ge=0, le=120)
    client_weight: Optional[float] = Field(None, description="Вес клиента (кг)", ge=0)
    
    # Симптомы и медицинская информация
    symptoms: Optional[str] = Field(None, description="Жалобы/симптомы клиента")
    symptoms_duration: Optional[str] = Field(None, description="Длительность симптомов")
    pain_character: Optional[str] = Field(None, description="Характер боли")
    visited_doctor: Optional[bool] = Field(None, description="Был ли у врача")
    has_contraindications: Optional[bool] = Field(None, description="Есть ли противопоказания")
    
    # Исследование
    study_type: Optional[str] = Field(
        None, 
        description="Тип исследования (МРТ головного мозга, комплекс, и т.д.)"
    )
    study_decision: Optional[str] = Field(
        None,
        description="Решение клиента: согласие на комплекс/часть/отказ"
    )
    
    # Цены и допродажи
    study_price: Optional[float] = Field(None, description="Стоимость исследования", ge=0)
    media_type: Optional[str] = Field(
        None,
        description="Формат заключения: бумажное/видеозаключение"
    )
    media_price: Optional[float] = Field(None, description="Стоимость носителя (диск, флешка)", ge=0)
    
    # Запись
    appointment_date: Optional[str] = Field(None, description="Дата записи")
    appointment_time: Optional[str] = Field(None, description="Время записи")
    
    # Адрес и контакты
    clinic_address: Optional[str] = Field(None, description="Адрес центра")
    
    # Метаданные
    has_discounts: Optional[bool] = Field(None, description="Применены ли скидки/льготы")
    
    @field_validator("client_phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """Валидирует формат телефона."""
        if v is None:
            return v
        
        # Удаляем всё кроме цифр
        digits = re.sub(r"\D", "", v)
        
        # Проверяем длину (10 или 11 цифр для РФ)
        if len(digits) not in (10, 11):
            raise ValueError("Телефон должен содержать 10 или 11 цифр")
        
        # Нормализуем к формату +7XXXXXXXXXX
        if len(digits) == 10:
            digits = "7" + digits
        elif digits[0] == "8":
            digits = "7" + digits[1:]
        
        return f"+{digits}"
    
    def is_complete(self) -> bool:
        """
        Проверяет, заполнены ли все критически важные слоты.
        
        Returns:
            True если можно завершить запись
        """
        required_fields = [
            self.client_name,
            self.client_phone,
            self.study_type,
            self.appointment_date,
            self.appointment_time,
            self.client_age,
            self.client_weight,
        ]
        
        return all(field is not None for field in required_fields)
    
    def to_sheets_row(self) -> dict:
        """
        Конвертирует слоты в формат для записи в Google Sheets.
        
        Returns:
            Словарь для Лист4
        """
        return {
            "timestamp": None,  # Заполнится при записи
            "client_name": self.client_name or "",
            "client_phone": self.client_phone or "",
            "client_age": self.client_age or "",
            "client_weight": self.client_weight or "",
            "symptoms": self.symptoms or "",
            "study_type": self.study_type or "",
            "appointment_date": self.appointment_date or "",
            "appointment_time": self.appointment_time or "",
            "study_price": self.study_price or "",
            "media_type": self.media_type or "",
            "media_price": self.media_price or "",
            "total_price": (self.study_price or 0) + (self.media_price or 0),
            "status": "записан",
        }


class SlotExtractionRequest(BaseModel):
    """Запрос на извлечение слотов из сообщения клиента."""
    
    user_message: str
    current_slots: Optional[DialogSlots] = None


class SlotExtractionResponse(BaseModel):
    """Ответ с извлечёнными слотами."""
    
    extracted_slots: DialogSlots
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в извлечении")

