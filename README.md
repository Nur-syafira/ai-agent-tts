# Sales Agent MVP

Локальный голосовой AI-агент для записи на МРТ с **минимальной латентностью** (280-640 мс рот→ухо).

## 🚀 Возможности

- ✅ **Потоковое ASR**: RealtimeSTT + faster-whisper large-v3-turbo (80-150 мс)
- ✅ **Silero VAD**: Endpointing + barge-in detection (100-200 мс)
- ✅ **LLM**: Qwen2.5-14B-Instruct-AWQ через vLLM (40-150 мс)
- ✅ **TTS**: Kokoro-82M (50-100 мс, английский) + Piper (80-150 мс, русский) + пререндер (<10 мс)
- ✅ **FSM**: 30 этапов диалога по скрипту МРТ
- ✅ **Google Sheets**: Append-only запись в Лист4
- ✅ **CUDA-only**: Guard-проверки GPU при старте
- ✅ **OpenTelemetry**: E2E латентность мониторинг

## 📊 Целевые метрики

| Компонент | Латентность | VRAM |
|-----------|-------------|------|
| ASR partial | 80-150 мс | ~3 GB |
| LLM inference | 40-150 мс | ~8 GB |
| TTS first-audio | 50-120 мс | ~1 GB |
| **E2E (рот→ухо)** | **250-600 мс** | **~12 GB** |

## 🛠️ Технологический стек

- **ASR**: RealtimeSTT + faster-whisper large-v3-turbo
- **VAD**: Silero VAD v5
- **LLM**: Qwen2.5-14B-Instruct-AWQ (vLLM)
- **TTS**: Piper + Kokoro-82M (опционально)
- **Policy**: LangGraph FSM + Pydantic slots
- **Storage**: Redis (сессии)
- **Sheets**: gspread-asyncio
- **Monitoring**: OpenTelemetry + Jaeger

## 📋 Требования

- **OS**: Linux (Ubuntu 22.04+)
- **Python**: 3.12
- **GPU**: CUDA-enabled (RTX 5090 рекомендуется, минимум 12 GB VRAM)
- **RAM**: 64 GB (рекомендуется)
- **CPU**: AMD Ryzen 9 9950X3D или аналог

## 🏗️ Архитектура

```
SIP/АТС → FreeSWITCH (mod_audio_fork)
   ├─(L16 PCM, 16 kHz, 160 ms)→ ASR Gateway
   │                              ↓
   │                         Policy Engine (FSM + Pydantic Slots)
   │                              ↓
   │                         LLM Service (vLLM)
   │                              ↓
   └←(PCM chunks, 200-300 ms)←  TTS Gateway
                                  ↓
                            Google Sheets Notifier (Лист4)
```

Подробная архитектура в [OVERVIEW.md](OVERVIEW.md).

## 🚀 Быстрый старт

### 1. Клонирование и установка

```bash
# Клонировать репо
git clone git@github.com:FUYOH666/AgentSales.git
cd AgentSales

# Создать venv
python3.12 -m venv venv
source venv/bin/activate

# Обновить pip
./venv/bin/python -m pip install --upgrade pip==25.2

# Установить зависимости
./venv/bin/pip install -r requirements.txt

# Скачать модели
./venv/bin/python scripts/download_models.py

# Установить Piper TTS
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
sudo mv piper/piper /usr/local/bin/
```

### 2. Конфигурация

```bash
# Скопировать .env
cp .env.example .env

# Отредактировать .env (укажи свои пути)
nano .env
```

### 3. Системные оптимизации

```bash
# GPU persistence + CPU performance mode
sudo ./scripts/setup_env.sh
```

### 4. Запуск сервисов

```bash
# Redis + Jaeger
docker-compose up -d redis jaeger

# vLLM (в отдельном терминале)
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --quantization awq \
  --enable-chunked-prefill \
  --enable-prefix-caching

# ASR Gateway (в отдельном терминале)
./venv/bin/python src/asr_gateway/main.py

# TTS Gateway (в отдельном терминале)
./venv/bin/python src/tts_gateway/main.py

# Policy Engine (в отдельном терминале)
./venv/bin/python src/policy_engine/main.py
```

### 5. Health Check

```bash
./venv/bin/python scripts/health_check.py
```

Ожидаемый вывод:

```
✅ Environment: Python 3.12.3, CUDA available, Google Credentials found
✅ GPU: RTX 5090 (32 GB)
✅ System: CPU 15%, RAM 40%, 38 GB available
✅ Redis: running
✅ Services:
   ASR Gateway.............. ✅ healthy
   LLM Service.............. ✅ healthy
   TTS Gateway.............. ✅ healthy
   Policy Engine............ ✅ healthy
```

## 📖 Документация

- [OVERVIEW.md](OVERVIEW.md) — архитектура и потоки данных
- [src/asr_gateway/README.md](src/asr_gateway/README.md) — ASR сервис
- [src/llm_service/README.md](src/llm_service/README.md) — LLM сервис
- [src/tts_gateway/README.md](src/tts_gateway/README.md) — TTS сервис
- [src/policy_engine/README.md](src/policy_engine/README.md) — Policy Engine
- [src/notifier/README.md](src/notifier/README.md) — Google Sheets интеграция
- [src/freeswitch_bridge/README.md](src/freeswitch_bridge/README.md) — FreeSWITCH интеграция

## 🧪 Тестирование

```bash
# Запустить все тесты
./venv/bin/pytest tests/ -v

# Тесты с покрытием
./venv/bin/pytest tests/ --cov=src --cov-report=html
```

## 📊 Мониторинг

### Jaeger UI

Откройте http://localhost:16686 для просмотра traces:

- E2E латентность (рот→ухо)
- ASR partial latency
- LLM TTFT (Time to First Token)
- TTS TTFA (Time to First Audio)

### Логи

Структурированные JSON-логи в stdout:

```json
{
  "timestamp": "2025-10-24T19:30:00.123Z",
  "level": "INFO",
  "service": "policy_engine",
  "message": "Dialog complete for session abc-123",
  "context": {"session_id": "abc-123", "slots_filled": 12}
}
```

## 🔒 Безопасность

- ✅ Credentials в `.gitignore`
- ✅ Service Account (не personal account)
- ✅ `.env` не коммитится
- ✅ Минимальные права доступа

## 🐛 Troubleshooting

### CUDA not available

```bash
# Проверить драйвер
nvidia-smi

# Проверить PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Переустановить PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Service not running

```bash
# Проверить порты
netstat -tuln | grep -E '800[0-3]'

# Проверить логи
./venv/bin/python src/asr_gateway/main.py  # Смотреть stdout
```

### Google Sheets permission denied

1. Открой таблицу
2. Share → добавь `client_email` из `credentials.json`
3. Дай права "Editor"

## 📝 Git workflow

```bash
# Создать feature ветку
git checkout -b feature/my-feature

# Коммитить с Conventional Commits
git commit -m "feat: add barge-in detection"

# Push и создать PR
git push origin feature/my-feature
```

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

Apache 2.0

## 👤 Author

**Aleksandr Mordvinov**  
Проект: ScanovichAI

## 🙏 Acknowledgments

- **Qwen Team** — Qwen2.5 LLM
- **Systran** — faster-whisper
- **Silero Team** — Silero VAD
- **Rhasspy** — Piper TTS
- **vLLM Team** — vLLM inference engine

---

Made with ❤️ for low-latency voice AI

