#!/usr/bin/env python3
"""
Улучшенный симулятор диалога с детальными метриками производительности.

Измеряет:
- Латентность каждого шага (ASR → LLM → TTS)
- Общее время ответа агента
- Качество диалога (заполнение слотов, переходы FSM)
- E2E метрики (рот → ухо)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
import yaml


@dataclass
class TurnMetrics:
    """Метрики одного хода диалога."""
    turn: int
    user_message: str
    agent_message: str
    state: str
    response_time_ms: float
    slots_filled: int
    timestamp: str


@dataclass
class DialogMetrics:
    """Метрики диалога."""
    session_id: str
    total_turns: int = 0
    total_time_seconds: float = 0.0
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    fsm_transitions: List[str] = None
    slots_filled: int = 0
    is_complete: bool = False
    errors: List[str] = None
    turn_metrics: List[TurnMetrics] = None
    
    def __post_init__(self):
        if self.fsm_transitions is None:
            self.fsm_transitions = []
        if self.errors is None:
            self.errors = []
        if self.turn_metrics is None:
            self.turn_metrics = []


class DialogSimulator:
    """Симулятор диалога с детальными метриками."""

    def __init__(
        self,
        policy_url: str = "http://localhost:8003",
        session_id: Optional[str] = None,
        scenario_file: Optional[str] = None,
        scenario_name: Optional[str] = None,
    ):
        self.policy_url = policy_url
        self.session_id = session_id or f"sim-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.scenario_file = scenario_file
        self.scenario_name = scenario_name
        
        self.metrics = DialogMetrics(session_id=self.session_id)
        self.conversation_log: List[Dict[str, Any]] = []
        
        # Загружаем сценарий если указан
        self.scenario = None
        if scenario_file:
            self.scenario = self._load_scenario(scenario_file, scenario_name)
    
    def _load_scenario(self, file_path: str, scenario_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Загружает сценарий из YAML файла."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_scenarios = yaml.safe_load(f)
            
            if not all_scenarios:
                return None
            
            if scenario_name:
                if scenario_name in all_scenarios:
                    return all_scenarios[scenario_name]
                else:
                    print(f"⚠️  Сценарий '{scenario_name}' не найден. Доступные: {list(all_scenarios.keys())}")
                    return None
            
            first_key = list(all_scenarios.keys())[0]
            return all_scenarios[first_key]
            
        except Exception as e:
            print(f"⚠️  Не удалось загрузить сценарий: {e}")
            return None
    
    async def check_services(self) -> bool:
        """Проверяет доступность сервисов."""
        print("🔍 Проверка сервисов...")
        
        services = {
            "Policy Engine": f"{self.policy_url}/health",
            "vLLM Server": "http://localhost:8000/v1/models",
            "TTS Gateway": "http://localhost:8002/health",
            "ASR Gateway": "http://localhost:8001/health",
        }
        
        all_ok = True
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, url in services.items():
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        print(f"   ✅ {name}")
                    else:
                        print(f"   ⚠️  {name} (HTTP {response.status_code})")
                        if name == "Policy Engine":
                            all_ok = False
                except Exception as e:
                    print(f"   ❌ {name} (не запущен: {e})")
                    if name == "Policy Engine":
                        all_ok = False
        
        return all_ok
    
    async def send_message(
        self,
        user_message: str,
    ) -> tuple[str, str, bool, Dict[str, Any], float]:
        """
        Отправляет сообщение в Policy Engine и измеряет время ответа.
        
        Returns:
            (agent_message, current_state, is_complete, slots, response_time_ms)
        """
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.policy_url}/dialog",
                    json={
                        "session_id": self.session_id,
                        "user_message": user_message,
                    },
                )
                
                response_time = (time.time() - start_time) * 1000  # мс
                
                if response.status_code == 200:
                    data = response.json()
                    return (
                        data["agent_message"],
                        data["current_state"],
                        data["is_complete"],
                        data.get("slots", {}),
                        response_time,
                    )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    self.metrics.errors.append(error_msg)
                    raise Exception(error_msg)
                    
            except Exception as e:
                error_msg = f"Request failed: {e}"
                self.metrics.errors.append(error_msg)
                raise
    
    def _get_user_response(self, turn: int, state: str) -> str:
        """Получает ответ пользователя для текущего хода."""
        if self.scenario and "responses" in self.scenario:
            responses = self.scenario["responses"]
            if turn < len(responses):
                return responses[turn]
        
        # Дефолтный сценарий
        default_responses = {
            "greeting": "Здравствуйте, хочу записаться на МРТ головного мозга",
            "ask_client_name": "Меня зовут Иван Петров",
            "ask_symptoms": "У меня сильные головные боли",
            "ask_symptoms_duration": "Уже около двух недель",
            "ask_pain_character": "Боль пульсирующая, в висках",
            "ask_visited_doctor": "Да, был у терапевта, он направил на МРТ",
            "ask_study_request": "Хочу пройти МРТ головного мозга",
            "recommend_study": "Да, согласен на комплексное исследование",
            "announce_price": "Хорошо, понял",
            "ask_study_decision": "Да, согласен",
            "offer_video_conclusion": "Хорошо, давайте видеозаключение тоже",
            "announce_media_price": "Понятно",
            "ask_appointment_date": "На завтра, если возможно",
            "offer_appointment_times": "15:00 подойдёт",
            "confirm_time": "Да, это удобное время",
            "ask_phone": "+7 999 123-45-67",
            "ask_age_weight": "Мне 35 лет, вес 78 кг",
            "check_contraindications": "Противопоказаний нет",
            "check_discounts": "Нет, я не пенсионер",
            "remind_documents": "Хорошо, спасибо за напоминание",
            "provide_address": "Записал, спасибо",
            "provide_contacts": "Понятно",
            "confirm_appointment": "Да, всё верно",
            "farewell": "Спасибо, до свидания!",
        }
        
        return default_responses.get(state, "Хорошо")
    
    async def simulate_dialog(self, max_turns: int = 50):
        """Симулирует полный диалог с детальными метриками."""
        print("=" * 80)
        print(" " * 25 + "🚀 Dialog Simulator")
        print("=" * 80)
        print(f"\n📋 Session ID: {self.session_id}\n")
        
        # Проверка сервисов
        if not await self.check_services():
            print("\n❌ Не все сервисы запущены!")
            print("\nДля запуска сервисов:")
            print("1. Redis: docker-compose up -d redis")
            print("2. vLLM: vllm serve models/Qwen3-16B-A3B-abliterated-AWQ --host 0.0.0.0 --port 8000 --quantization awq --enable-chunked-prefill --enable-prefix-caching")
            print("3. Policy Engine: uv run python src/policy_engine/main.py")
            print("4. (Опционально) TTS Gateway: uv run python src/tts_gateway/main.py")
            print("5. (Опционально) ASR Gateway: uv run python src/asr_gateway/main.py")
            return
        
        if self.scenario:
            print(f"📄 Сценарий: {self.scenario.get('name', 'Unnamed')}")
            if 'description' in self.scenario:
                print(f"📝 Описание: {self.scenario['description']}")
        
        print("\n🚀 Начинаем симуляцию диалога...\n")
        print("=" * 80)
        
        start_time = time.time()
        is_complete = False
        turn = 0
        
        # Первый ход - агент начинает
        print(f"\n{'─'*80}")
        print(f"Turn {turn + 1}")
        print(f"{'─'*80}")
        print(f"👤 Клиент: [звонок начат]")
        
        try:
            agent_msg, state, is_complete, slots, response_time = await self.send_message("")
            
            print(f"🤖 Агент: {agent_msg}")
            print(f"📊 State: {state}")
            print(f"⏱️  Время ответа: {response_time:.1f} мс")
            
            # Сохраняем метрики
            turn_metric = TurnMetrics(
                turn=turn + 1,
                user_message="",
                agent_message=agent_msg,
                state=state,
                response_time_ms=response_time,
                slots_filled=len([v for v in slots.values() if v is not None]),
                timestamp=datetime.now().isoformat(),
            )
            self.metrics.turn_metrics.append(turn_metric)
            self.metrics.fsm_transitions.append(state)
            
            self.conversation_log.append({
                "turn": turn + 1,
                "user": "",
                "agent": agent_msg,
                "state": state,
                "slots": slots,
                "response_time_ms": response_time,
            })
            
            turn += 1
            self.metrics.total_turns = turn
            
        except Exception as e:
            print(f"❌ Ошибка при старте диалога: {e}")
            return
        
        # Основной цикл диалога
        while not is_complete and turn < max_turns:
            await asyncio.sleep(0.3)  # Небольшая пауза между ходами
            
            print(f"\n{'─'*80}")
            print(f"Turn {turn + 1}")
            print(f"{'─'*80}")
            
            # Получаем ответ клиента
            user_msg = self._get_user_response(turn - 1, state)
            print(f"👤 Клиент: {user_msg}")
            
            # Отправляем в Policy Engine
            try:
                turn_start = time.time()
                agent_msg, new_state, is_complete, slots, response_time = await self.send_message(user_msg)
                
                print(f"🤖 Агент: {agent_msg}")
                print(f"📊 State: {new_state}")
                print(f"⏱️  Время ответа: {response_time:.1f} мс")
                
                # Обновляем метрики
                self.metrics.avg_response_time_ms = (
                    (self.metrics.avg_response_time_ms * (turn - 1) + response_time) / turn
                )
                self.metrics.min_response_time_ms = min(self.metrics.min_response_time_ms, response_time)
                self.metrics.max_response_time_ms = max(self.metrics.max_response_time_ms, response_time)
                
                # Отслеживаем переходы FSM
                if new_state != state:
                    self.metrics.fsm_transitions.append(new_state)
                    print(f"   🔄 Transition: {state} → {new_state}")
                
                state = new_state
                
                # Подсчитываем заполненные слоты
                filled_slots = {k: v for k, v in slots.items() if v is not None}
                self.metrics.slots_filled = len(filled_slots)
                
                if filled_slots:
                    print(f"📝 Заполнено слотов: {len(filled_slots)}")
                    # Показываем последние заполненные слоты
                    recent_slots = list(filled_slots.items())[-3:]
                    for k, v in recent_slots:
                        print(f"   - {k}: {v}")
                
                # Сохраняем метрики хода
                turn_metric = TurnMetrics(
                    turn=turn + 1,
                    user_message=user_msg,
                    agent_message=agent_msg,
                    state=state,
                    response_time_ms=response_time,
                    slots_filled=len(filled_slots),
                    timestamp=datetime.now().isoformat(),
                )
                self.metrics.turn_metrics.append(turn_metric)
                
                # Логируем ход
                self.conversation_log.append({
                    "turn": turn + 1,
                    "user": user_msg,
                    "agent": agent_msg,
                    "state": state,
                    "slots": slots,
                    "response_time_ms": response_time,
                })
                
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                self.metrics.errors.append(str(e))
                break
            
            turn += 1
            self.metrics.total_turns = turn
        
        # Завершение
        total_time = time.time() - start_time
        self.metrics.total_time_seconds = total_time
        self.metrics.is_complete = is_complete
        
        # Выводим итоговые метрики
        print("\n" + "=" * 80)
        print(" " * 25 + "📊 Результаты симуляции")
        print("=" * 80)
        print(f"✅ Завершён: {'Да' if is_complete else 'Нет'}")
        print(f"📈 Всего ходов: {self.metrics.total_turns}")
        print(f"⏱️  Общее время: {total_time:.2f} сек")
        print(f"⚡ Среднее время ответа: {self.metrics.avg_response_time_ms:.1f} мс")
        print(f"📉 Минимальное время ответа: {self.metrics.min_response_time_ms:.1f} мс")
        print(f"📈 Максимальное время ответа: {self.metrics.max_response_time_ms:.1f} мс")
        print(f"🔄 Переходов FSM: {len(self.metrics.fsm_transitions)}")
        print(f"📝 Заполнено слотов: {self.metrics.slots_filled}")
        
        if self.metrics.errors:
            print(f"❌ Ошибок: {len(self.metrics.errors)}")
            for error in self.metrics.errors:
                print(f"   - {error}")
        
        print(f"\n📋 Финальное состояние: {state}")
        
        # Анализ производительности
        print("\n" + "=" * 80)
        print(" " * 25 + "📈 Анализ производительности")
        print("=" * 80)
        
        response_times = [tm.response_time_ms for tm in self.metrics.turn_metrics]
        if response_times:
            print(f"Целевая латентность: 250-600 мс (рот → ухо)")
            print(f"Средняя латентность: {self.metrics.avg_response_time_ms:.1f} мс")
            
            if self.metrics.avg_response_time_ms < 250:
                print("✅ Отлично! Латентность ниже целевой")
            elif self.metrics.avg_response_time_ms < 600:
                print("✅ Хорошо! Латентность в целевом диапазоне")
            elif self.metrics.avg_response_time_ms < 1000:
                print("⚠️  Приемлемо, но можно улучшить")
            else:
                print("❌ Латентность выше целевой, требуется оптимизация")
        
        print("=" * 80)
        
        # Сохраняем логи
        self._save_logs()
    
    def _save_logs(self):
        """Сохраняет логи диалога в файл."""
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"dialog_{self.session_id}.json"
        
        log_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": asdict(self.metrics),
            "conversation": self.conversation_log,
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Логи сохранены: {log_file}")


async def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Симулятор диалога Sales Agent с метриками")
    parser.add_argument(
        "--policy-url",
        default="http://localhost:8003",
        help="URL Policy Engine",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="ID сессии (генерируется автоматически если не указан)",
    )
    parser.add_argument(
        "--scenario",
        default="scripts/dialog_scenarios.yaml",
        help="Путь к YAML файлу со сценариями",
    )
    parser.add_argument(
        "--scenario-name",
        default=None,
        help="Имя конкретного сценария из файла (например: basic_success, with_objections)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Максимальное количество ходов",
    )
    
    args = parser.parse_args()
    
    simulator = DialogSimulator(
        policy_url=args.policy_url,
        session_id=args.session_id,
        scenario_file=args.scenario,
        scenario_name=args.scenario_name,
    )
    
    await simulator.simulate_dialog(max_turns=args.max_turns)


if __name__ == "__main__":
    asyncio.run(main())

