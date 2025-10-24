#!/usr/bin/env python3

"""
Симулятор диалога между AI-агентом и пациентом.

Автоматически проходит через все этапы диалога записи на МРТ.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import httpx
from datetime import datetime
import random


class DialogSimulator:
    """Симулятор диалога."""

    def __init__(
        self,
        policy_url: str = "http://localhost:8003",
        tts_url: str = "http://localhost:8002",
        session_id: str = None,
    ):
        """
        Инициализация.
        
        Args:
            policy_url: URL Policy Engine
            tts_url: URL TTS Gateway
            session_id: ID сессии
        """
        self.policy_url = policy_url
        self.tts_url = tts_url
        self.session_id = session_id or f"sim-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Сценарий диалога (ответы "пациента")
        self.patient_responses = [
            "Здравствуйте, хочу записаться на МРТ головного мозга",
            "Меня зовут Иван Петров",
            "У меня сильные головные боли",
            "Уже около двух недель",
            "Боль пульсирующая, в висках",
            "Да, был у терапевта, он направил на МРТ",
            "Хочу пройти МРТ головного мозга",
            "Да, согласен на комплексное исследование",
            "Хорошо, давайте видеозаключение тоже",
            "На завтра, если возможно",
            "15:00 подойдёт",
            "Да, это удобное время",
            "+7 999 123-45-67",
            "Мне 35 лет, вес 78 кг",
            "Противопоказаний нет",
            "Нет, я не пенсионер",
            "Хорошо, спасибо за напоминание",
            "Москва, улица Ленина",
            "Записал, спасибо",
            "Спасибо, до свидания!",
        ]
        
        self.turn = 0

    async def check_services(self):
        """Проверяет сервисы."""
        print("🔍 Проверка сервисов...")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.policy_url}/health")
                if response.status_code == 200:
                    print("   ✅ Policy Engine")
                else:
                    print(f"   ❌ Policy Engine (HTTP {response.status_code})")
                    return False
            except:
                print("   ❌ Policy Engine (не запущен)")
                return False
            
            try:
                response = await client.get(f"{self.tts_url}/health")
                if response.status_code == 200:
                    print("   ✅ TTS Gateway")
                else:
                    print(f"   ⚠️  TTS Gateway (HTTP {response.status_code})")
                    # TTS опционален для симуляции
            except:
                print("   ⚠️  TTS Gateway (не запущен, будет пропущен)")
        
        return True

    async def send_message(self, text: str):
        """
        Отправляет сообщение в Policy Engine.
        
        Args:
            text: Текст пациента
            
        Returns:
            Ответ агента, состояние, завершён ли диалог
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.policy_url}/dialog",
                json={
                    "session_id": self.session_id,
                    "user_message": text,
                },
            )
            
            if response.status_code == 200:
                data = response.json()
                return (
                    data["agent_message"],
                    data["current_state"],
                    data["is_complete"],
                    data.get("slots", {}),
                )
            else:
                raise Exception(f"Policy Engine error: HTTP {response.status_code}")

    async def simulate_dialog(self):
        """Симулирует полный диалог."""
        print("=" * 80)
        print(" " * 30 + "Dialog Simulator")
        print("=" * 80)
        print(f"\n📋 Session ID: {self.session_id}\n")
        
        # Проверка сервисов
        if not await self.check_services():
            print("\n❌ Policy Engine не запущен!")
            return
        
        print("\n🚀 Начинаем симуляцию диалога...\n")
        
        is_complete = False
        response_idx = 0
        
        # Первый ход - агент начинает
        print(f"{'='*80}")
        print(f"Turn {self.turn + 1}")
        print(f"{'='*80}")
        print(f"👤 Пациент: [звонок начат]")
        
        agent_msg, state, is_complete, slots = await self.send_message("")
        
        print(f"🤖 Агент: {agent_msg}")
        print(f"📊 State: {state}")
        print()
        
        self.turn += 1
        
        while not is_complete and response_idx < len(self.patient_responses):
            await asyncio.sleep(1)  # Пауза между сообщениями
            
            print(f"{'='*80}")
            print(f"Turn {self.turn + 1}")
            print(f"{'='*80}")
            
            # Ответ пациента
            patient_msg = self.patient_responses[response_idx]
            print(f"👤 Пациент: {patient_msg}")
            
            # Ответ агента
            try:
                agent_msg, state, is_complete, slots = await self.send_message(patient_msg)
                
                print(f"🤖 Агент: {agent_msg}")
                print(f"📊 State: {state}")
                
                # Показываем заполненные слоты
                filled_slots = {k: v for k, v in slots.items() if v is not None}
                if filled_slots:
                    print(f"📝 Заполнено слотов: {len(filled_slots)}")
                    for k, v in list(filled_slots.items())[:5]:  # Показываем первые 5
                        print(f"   - {k}: {v}")
                
                print()
                
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                break
            
            response_idx += 1
            self.turn += 1
        
        print("=" * 80)
        if is_complete:
            print("✅ Диалог успешно завершён!")
            print(f"   Всего turns: {self.turn}")
            print(f"   Финальное состояние: {state}")
        else:
            print("⚠️  Диалог не завершён (закончились ответы пациента)")
        print("=" * 80)


async def main():
    """Главная функция."""
    simulator = DialogSimulator()
    await simulator.simulate_dialog()


if __name__ == "__main__":
    asyncio.run(main())

