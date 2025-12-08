"""
Пример кастомизации FSM для своего домена.

Этот пример показывает, как создать свой FSM с кастомными состояниями.
"""

import sys
from pathlib import Path
from enum import Enum
from typing import Optional
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy_engine.fsm import DialogFSM, FSMTransition
from src.policy_engine.slots import DialogSlots


class CustomState(str, Enum):
    """Кастомные состояния для вашего домена."""
    
    START = "start"
    GREETING = "greeting"
    COLLECT_INFO = "collect_info"
    CONFIRM = "confirm"
    COMPLETE = "complete"
    END = "end"


class CustomFSM(DialogFSM):
    """Кастомный FSM для вашего домена."""
    
    def __init__(self):
        # Инициализация с кастомными состояниями
        super().__init__()
        # Переопределить current_state если нужно
        # self.current_state = CustomState.START
    
    def _build_transitions(self) -> list[FSMTransition]:
        """Построить граф переходов для вашего домена."""
        return [
            # Пример переходов
            FSMTransition(CustomState.START, CustomState.GREETING),
            FSMTransition(CustomState.GREETING, CustomState.COLLECT_INFO),
            FSMTransition(
                CustomState.COLLECT_INFO,
                CustomState.CONFIRM,
                condition=lambda slots, msg: self._has_required_info(slots)
            ),
            FSMTransition(CustomState.CONFIRM, CustomState.COMPLETE),
            FSMTransition(CustomState.COMPLETE, CustomState.END),
        ]
    
    def _has_required_info(self, slots: DialogSlots) -> bool:
        """Проверка, что собрана необходимая информация."""
        # Ваша логика проверки
        return bool(slots.client_name and slots.client_phone)


# Пример использования
if __name__ == "__main__":
    fsm = CustomFSM()
    print(f"Текущее состояние: {fsm.current_state}")
    print(f"Переходы: {len(fsm.transitions)}")

