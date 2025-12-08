"""
Finite State Machine для диалога записи на МРТ.

Основано на script_evaluation_type_A.md (30 этапов).
"""

from enum import Enum
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging

from src.policy_engine.slots import DialogSlots


class DialogState(str, Enum):
    """Состояния FSM для диалога записи на МРТ."""
    
    # Начало диалога
    START = "start"
    GREETING = "greeting"  # 1, 2, 3: Приветствие, представление, название центра
    
    # Сбор информации о клиенте
    ASK_CLIENT_NAME = "ask_client_name"  # 4: Имя клиента
    ASK_SYMPTOMS = "ask_symptoms"  # 5: Жалобы/симптомы
    ASK_SYMPTOMS_DURATION = "ask_symptoms_duration"  # 6: Длительность симптомов
    ASK_PAIN_CHARACTER = "ask_pain_character"  # 7: Характер боли
    ASK_VISITED_DOCTOR = "ask_visited_doctor"  # 8: Был ли у врача
    
    # Определение исследования
    ASK_STUDY_REQUEST = "ask_study_request"  # 9: Что именно хочет клиент
    RECOMMEND_STUDY = "recommend_study"  # 10, 11: Рекомендация + аргументы
    ANNOUNCE_PRICE = "announce_price"  # 12: Стоимость
    ASK_STUDY_DECISION = "ask_study_decision"  # 13: Выбор клиента
    
    # Допродажи
    OFFER_VIDEO_CONCLUSION = "offer_video_conclusion"  # 14, 15: Видеозаключение + описание
    ANNOUNCE_MEDIA_PRICE = "announce_media_price"  # 20: Стоимость носителя
    
    # Запись
    ASK_APPOINTMENT_DATE = "ask_appointment_date"  # 16: Дата
    OFFER_APPOINTMENT_TIMES = "offer_appointment_times"  # 17: Варианты времени
    CONFIRM_TIME = "confirm_time"  # 25: Уточнение удобства времени
    
    # Персональные данные
    ASK_PHONE = "ask_phone"  # 18: Телефон
    ASK_AGE_WEIGHT = "ask_age_weight"  # 18: Возраст, вес
    
    # Дополнительная информация
    CHECK_CONTRAINDICATIONS = "check_contraindications"  # 23: Противопоказания
    CHECK_DISCOUNTS = "check_discounts"  # 26: Скидки и льготы
    REMIND_DOCUMENTS = "remind_documents"  # 19: Документы и подготовка
    
    # Завершение
    PROVIDE_ADDRESS = "provide_address"  # 29: Адрес центра
    PROVIDE_CONTACTS = "provide_contacts"  # 30: Телефон для связи
    CONFIRM_APPOINTMENT = "confirm_appointment"  # 24: Итоговое подтверждение
    FAREWELL = "farewell"  # Прощание
    
    # Служебные
    END = "end"
    ERROR = "error"


@dataclass
class FSMTransition:
    """Переход между состояниями FSM."""
    
    from_state: DialogState
    to_state: DialogState
    condition: Optional[Callable[[DialogSlots, str], bool]] = None
    priority: int = 0  # Для разрешения конфликтов


class DialogFSM:
    """Finite State Machine для управления диалогом."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Инициализация FSM.
        
        Args:
            logger: Logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.current_state = DialogState.START
        self.transitions = self._build_transitions()
        
    def _build_transitions(self) -> list[FSMTransition]:
        """
        Строит граф переходов FSM.
        
        Returns:
            Список переходов
        """
        return [
            # Начало диалога
            FSMTransition(DialogState.START, DialogState.GREETING),
            FSMTransition(DialogState.GREETING, DialogState.ASK_CLIENT_NAME),
            
            # Сбор информации о клиенте
            FSMTransition(
                DialogState.ASK_CLIENT_NAME,
                DialogState.ASK_SYMPTOMS,
                condition=lambda slots, msg: slots.client_name is not None,
            ),
            FSMTransition(
                DialogState.ASK_SYMPTOMS,
                DialogState.ASK_SYMPTOMS_DURATION,
                condition=lambda slots, msg: slots.symptoms is not None,
            ),
            FSMTransition(
                DialogState.ASK_SYMPTOMS_DURATION,
                DialogState.ASK_PAIN_CHARACTER,
                condition=lambda slots, msg: slots.symptoms_duration is not None,
            ),
            FSMTransition(
                DialogState.ASK_PAIN_CHARACTER,
                DialogState.ASK_VISITED_DOCTOR,
                condition=lambda slots, msg: slots.pain_character is not None,
            ),
            FSMTransition(
                DialogState.ASK_VISITED_DOCTOR,
                DialogState.ASK_STUDY_REQUEST,
                condition=lambda slots, msg: slots.visited_doctor is not None,
            ),
            
            # Определение исследования
            FSMTransition(
                DialogState.ASK_STUDY_REQUEST,
                DialogState.RECOMMEND_STUDY,
            ),
            FSMTransition(
                DialogState.RECOMMEND_STUDY,
                DialogState.ANNOUNCE_PRICE,
            ),
            FSMTransition(
                DialogState.ANNOUNCE_PRICE,
                DialogState.ASK_STUDY_DECISION,
            ),
            FSMTransition(
                DialogState.ASK_STUDY_DECISION,
                DialogState.OFFER_VIDEO_CONCLUSION,
                condition=lambda slots, msg: slots.study_decision is not None,
            ),
            
            # Допродажи
            FSMTransition(
                DialogState.OFFER_VIDEO_CONCLUSION,
                DialogState.ANNOUNCE_MEDIA_PRICE,
            ),
            FSMTransition(
                DialogState.ANNOUNCE_MEDIA_PRICE,
                DialogState.ASK_APPOINTMENT_DATE,
            ),
            
            # Запись
            FSMTransition(
                DialogState.ASK_APPOINTMENT_DATE,
                DialogState.OFFER_APPOINTMENT_TIMES,
                condition=lambda slots, msg: slots.appointment_date is not None,
            ),
            FSMTransition(
                DialogState.OFFER_APPOINTMENT_TIMES,
                DialogState.CONFIRM_TIME,
                condition=lambda slots, msg: slots.appointment_time is not None,
            ),
            FSMTransition(
                DialogState.CONFIRM_TIME,
                DialogState.ASK_PHONE,
            ),
            
            # Персональные данные
            FSMTransition(
                DialogState.ASK_PHONE,
                DialogState.ASK_AGE_WEIGHT,
                condition=lambda slots, msg: slots.client_phone is not None,
            ),
            FSMTransition(
                DialogState.ASK_AGE_WEIGHT,
                DialogState.CHECK_CONTRAINDICATIONS,
                condition=lambda slots, msg: (
                    slots.client_age is not None and slots.client_weight is not None
                ),
            ),
            
            # Дополнительная информация
            FSMTransition(
                DialogState.CHECK_CONTRAINDICATIONS,
                DialogState.CHECK_DISCOUNTS,
            ),
            FSMTransition(
                DialogState.CHECK_DISCOUNTS,
                DialogState.REMIND_DOCUMENTS,
            ),
            FSMTransition(
                DialogState.REMIND_DOCUMENTS,
                DialogState.PROVIDE_ADDRESS,
            ),
            FSMTransition(
                DialogState.PROVIDE_ADDRESS,
                DialogState.PROVIDE_CONTACTS,
            ),
            FSMTransition(
                DialogState.PROVIDE_CONTACTS,
                DialogState.CONFIRM_APPOINTMENT,
            ),
            
            # Завершение
            FSMTransition(
                DialogState.CONFIRM_APPOINTMENT,
                DialogState.FAREWELL,
            ),
            FSMTransition(
                DialogState.FAREWELL,
                DialogState.END,
            ),
        ]

    def get_next_state(
        self, slots: DialogSlots, user_message: str
    ) -> Optional[DialogState]:
        """
        Определяет следующее состояние на основе текущего состояния и слотов.
        
        Args:
            slots: Текущие слоты диалога
            user_message: Последнее сообщение пользователя
            
        Returns:
            Следующее состояние или None если нет валидного перехода
        """
        # Находим все возможные переходы из текущего состояния
        possible_transitions = [
            t for t in self.transitions if t.from_state == self.current_state
        ]
        
        if not possible_transitions:
            self.logger.warning(f"No transitions from state {self.current_state}")
            return None
        
        # Проверяем условия переходов
        for transition in sorted(possible_transitions, key=lambda t: -t.priority):
            if transition.condition is None or transition.condition(slots, user_message):
                self.logger.debug(
                    f"Transition: {self.current_state} -> {transition.to_state}"
                )
                return transition.to_state
        
        # Если ни одно условие не выполнено, остаёмся в текущем состоянии
        self.logger.debug(f"No valid transition from {self.current_state}, staying")
        return self.current_state

    def transition(self, new_state: DialogState):
        """
        Выполняет переход в новое состояние.
        
        Args:
            new_state: Новое состояние
        """
        self.logger.info(f"FSM: {self.current_state.value} -> {new_state.value}")
        self.current_state = new_state

    def get_state_context(self, state: Optional[DialogState] = None) -> Dict[str, Any]:
        """
        Возвращает контекст для текущего состояния.
        
        Args:
            state: Состояние (по умолчанию текущее)
            
        Returns:
            Словарь с контекстом для генерации ответа
        """
        state = state or self.current_state
        
        # Контекст для каждого состояния
        contexts = {
            DialogState.GREETING: {
                "center_name": "Медицинский центр МРТ 1.5Т",
                "admin_name": "Администратор",
            },
            DialogState.ANNOUNCE_PRICE: {
                "mri_brain_price": 4500,
                "mri_complex_price": 7500,
                "discount_for_complex": "экономия 1500 рублей",
            },
            DialogState.ANNOUNCE_MEDIA_PRICE: {
                "disk_price": 300,
                "usb_price": 500,
            },
            DialogState.OFFER_APPOINTMENT_TIMES: {
                "available_times": ["09:00", "12:00", "15:00", "18:00"],
            },
            DialogState.PROVIDE_ADDRESS: {
                "address": "г. Москва, ул. Примерная, д. 10",
            },
            DialogState.PROVIDE_CONTACTS: {
                "phone": "+7 (495) 123-45-67",
                "work_hours": "с 8:00 до 21:00",
            },
        }
        
        return contexts.get(state, {})

    def reset(self):
        """Сбрасывает FSM в начальное состояние."""
        self.logger.info("Resetting FSM to START state")
        self.current_state = DialogState.START

    def is_terminal_state(self) -> bool:
        """
        Проверяет, находится ли FSM в терминальном состоянии.
        
        Returns:
            True если диалог завершён
        """
        return self.current_state in (DialogState.END, DialogState.ERROR)

