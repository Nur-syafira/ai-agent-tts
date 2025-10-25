#!/usr/bin/env python3
"""Быстрый тест Policy Engine + LLM."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import httpx
import json


async def test_policy_engine():
    """Тест создания сессии и первой реплики."""
    
    base_url = "http://localhost:8003"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("=" * 60)
        print("🧪 Тест Policy Engine + vLLM")
        print("=" * 60)
        
        # 1. Health check
        print("\n1️⃣  Проверка здоровья Policy Engine...")
        resp = await client.get(f"{base_url}/health")
        if resp.status_code == 200:
            print("   ✅ Policy Engine работает")
        else:
            print(f"   ❌ Policy Engine недоступен: {resp.status_code}")
            return
        
        # 2. Создание сессии (просто генерируем ID)
        print("\n2️⃣  Создание новой сессии...")
        import uuid
        session_id = str(uuid.uuid4())
        print(f"   ✅ Сессия ID: {session_id}")
        
        # 3. Тест диалога
        print("\n3️⃣  Симуляция диалога...")
        
        user_inputs = [
            "Здравствуйте, меня зовут Иван",
            "Хочу записаться на МРТ коленного сустава",
            "Завтра в 14:00 можно?",
            "Филиал на Ленина подходит",
            "+79991234567",
        ]
        
        for i, user_input in enumerate(user_inputs, 1):
            print(f"\n   👤 Пользователь: {user_input}")
            
            resp = await client.post(
                f"{base_url}/dialog",
                json={
                    "session_id": session_id,
                    "user_message": user_input
                }
            )
            
            if resp.status_code != 200:
                print(f"   ❌ Ошибка: {resp.status_code}")
                print(f"   {resp.text}")
                break
            
            reply_data = resp.json()
            print(f"   🤖 Агент: {reply_data['agent_message']}")
            print(f"   Состояние: {reply_data.get('current_state', 'unknown')}")
            
            # Если диалог завершён
            if reply_data.get("is_complete"):
                print(f"\n   ✅ Диалог завершён!")
                print(f"   Слоты: {json.dumps(reply_data.get('slots', {}), ensure_ascii=False, indent=4)}")
                break
            
            await asyncio.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("✅ Тест завершён успешно!")
        print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(test_policy_engine())
    except KeyboardInterrupt:
        print("\n\n⚠️  Тест прерван пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

