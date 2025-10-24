#!/usr/bin/env python3

"""
Тест голосового диалога через микрофон.

Использует микрофон компьютера для записи речи, распознаёт через ASR,
отправляет в Policy Engine, получает ответ и воспроизводит через TTS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


class VoiceChatTester:
    """Тестер голосового диалога."""

    def __init__(
        self,
        asr_url: str = "http://localhost:8001",
        policy_url: str = "http://localhost:8003",
        tts_url: str = "http://localhost:8002",
        session_id: str = None,
    ):
        """
        Инициализация.
        
        Args:
            asr_url: URL ASR Gateway
            policy_url: URL Policy Engine
            tts_url: URL TTS Gateway
            session_id: ID сессии (генерируется автоматически если None)
        """
        self.asr_url = asr_url
        self.policy_url = policy_url
        self.tts_url = tts_url
        self.session_id = session_id or f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self.sample_rate = 16000  # 16 kHz для ASR
        self.channels = 1  # Mono
        
        print(f"🎤 Session ID: {self.session_id}")

    async def check_services(self):
        """Проверяет доступность сервисов."""
        print("\n🔍 Проверка сервисов...")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            services = {
                "ASR Gateway": f"{self.asr_url}/health",
                "Policy Engine": f"{self.policy_url}/health",
                "TTS Gateway": f"{self.tts_url}/health",
            }
            
            all_ok = True
            for name, url in services.items():
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        print(f"   ✅ {name}")
                    else:
                        print(f"   ❌ {name} (HTTP {response.status_code})")
                        all_ok = False
                except Exception as e:
                    print(f"   ❌ {name} (не запущен)")
                    all_ok = False
            
            return all_ok

    def record_audio(self, duration: float = 5.0):
        """
        Записывает аудио с микрофона.
        
        Args:
            duration: Длительность записи (секунды)
            
        Returns:
            Numpy array с аудио
        """
        print(f"\n🎤 Запись {duration} секунд...")
        print("   Говорите!")
        
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
        )
        sd.wait()
        
        print("   ✅ Запись завершена")
        return audio.flatten()

    async def transcribe_audio(self, audio: np.ndarray):
        """
        Распознаёт аудио через ASR Gateway.
        
        Args:
            audio: Аудио данные
            
        Returns:
            Текст транскрипта
        """
        print("\n📝 Распознавание речи...")
        
        # TODO: Реализовать WebSocket клиент для ASR
        # Пока заглушка - возвращаем mock
        print("   ⚠️  ASR WebSocket не реализован в этом тесте")
        print("   Введите текст вручную:")
        text = input("   > ")
        return text

    async def send_to_policy(self, text: str):
        """
        Отправляет текст в Policy Engine.
        
        Args:
            text: Текст пользователя
            
        Returns:
            Ответ агента
        """
        print("\n🤖 Обработка в Policy Engine...")
        
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
                agent_message = data["agent_message"]
                current_state = data["current_state"]
                is_complete = data["is_complete"]
                
                print(f"   State: {current_state}")
                print(f"   Response: {agent_message}")
                
                if is_complete:
                    print("   ✅ Диалог завершён!")
                
                return agent_message, is_complete
            else:
                print(f"   ❌ Ошибка: HTTP {response.status_code}")
                return None, False

    async def synthesize_speech(self, text: str):
        """
        Синтезирует речь через TTS Gateway.
        
        Args:
            text: Текст для синтеза
            
        Returns:
            Аудио данные
        """
        print("\n🔊 Синтез речи...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.tts_url}/synthesize",
                json={"text": text, "use_fallback": False},
            )
            
            if response.status_code == 200:
                # Получаем аудио
                audio_bytes = response.content
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Получаем sample rate из headers
                sample_rate = int(response.headers.get("X-Sample-Rate", 24000))
                
                print(f"   ✅ Синтез завершён ({len(audio)} samples, {sample_rate} Hz)")
                return audio, sample_rate
            else:
                print(f"   ❌ Ошибка: HTTP {response.status_code}")
                return None, None

    def play_audio(self, audio: np.ndarray, sample_rate: int):
        """
        Воспроизводит аудио через динамики.
        
        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизации
        """
        print("\n🔊 Воспроизведение...")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        print("   ✅ Воспроизведение завершено")

    async def run_dialog(self):
        """Запускает диалог."""
        print("=" * 70)
        print(" " * 20 + "Voice Chat Test")
        print("=" * 70)
        
        # Проверка сервисов
        if not await self.check_services():
            print("\n❌ Не все сервисы запущены. Запусти их и повтори.")
            return
        
        print("\n✅ Все сервисы готовы!")
        print()
        print("Инструкции:")
        print("  - Введи текст (вместо записи с микрофона)")
        print("  - Агент ответит через TTS")
        print("  - Введи 'quit' для выхода")
        print()
        
        is_complete = False
        turn = 0
        
        while not is_complete:
            turn += 1
            print(f"\n{'='*70}")
            print(f"Turn {turn}")
            print(f"{'='*70}")
            
            # 1. Записываем аудио (или вводим текст)
            # audio = self.record_audio(duration=5.0)
            
            # 2. Распознаём речь
            user_text = await self.transcribe_audio(None)
            
            if user_text.lower() == "quit":
                print("\n👋 Выход...")
                break
            
            # 3. Отправляем в Policy Engine
            agent_text, is_complete = await self.send_to_policy(user_text)
            
            if not agent_text:
                continue
            
            # 4. Синтезируем речь
            audio, sample_rate = await self.synthesize_speech(agent_text)
            
            if audio is not None:
                # 5. Воспроизводим
                self.play_audio(audio, sample_rate)
        
        print("\n" + "="*70)
        print("✅ Диалог завершён!")
        print("="*70)


async def main():
    """Главная функция."""
    tester = VoiceChatTester()
    await tester.run_dialog()


if __name__ == "__main__":
    asyncio.run(main())

