# Инструкция по установке Sales Agent MVP

## 1. Настройка Git

```bash
# Укажите ваши данные (замените на свои)
git config user.email "your.email@example.com"
git config user.name "Your Name"

# Сделайте первый коммит
git commit -m "feat: initial Sales Agent MVP implementation

- ASR Gateway (RealtimeSTT + faster-whisper large-v3-turbo + Silero VAD)
- LLM Service (vLLM wrapper for Qwen2.5-14B-Instruct-AWQ)
- TTS Gateway (Piper + Kokoro-82M stub + prerender cache)
- Policy Engine (LangGraph FSM with 30 states + Pydantic slots)
- Google Sheets Notifier (async append-only to Лист4)
- FreeSWITCH Bridge stub
- Shared utilities (logging, config, health, metrics)
- Scripts (setup_env.sh, download_models.py, health_check.py)
- Full documentation (README.md, OVERVIEW.md)
- Docker Compose for Redis + Jaeger

Target E2E latency: 280-640 ms
Hardware: RTX 5090 + Ryzen 9950X3D"
```

## 2. Создание виртуального окружения

```bash
python3.12 -m venv venv
source venv/bin/activate
./venv/bin/python -m pip install --upgrade pip==25.2
```

## 3. Установка зависимостей

```bash
./venv/bin/pip install -r requirements.txt
```

## 4. Установка Piper TTS

```bash
# Скачать Piper
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
sudo mv piper/piper /usr/local/bin/

# Проверить установку
piper --version
```

## 5. Скачивание моделей

```bash
./venv/bin/python scripts/download_models.py
```

Это скачает:
- faster-whisper large-v3-turbo
- Qwen2.5-14B-Instruct-AWQ
- Piper TTS ru_RU-dmitri-medium
- Silero VAD

## 6. Конфигурация

```bash
# Скопировать .env
cp .env.example .env

# Отредактировать .env
nano .env
```

Убедись что указаны правильные пути:
- `GOOGLE_CREDENTIALS_PATH` — путь к credentials.json
- `GOOGLE_SHEET_ID` — ID твоей Google таблицы
- `CUDA_VISIBLE_DEVICES=0` — GPU ID

## 7. Системные оптимизации

```bash
sudo ./scripts/setup_env.sh
```

Это настроит:
- GPU persistence mode
- CPU governor = performance

## 8. Запуск Redis и Jaeger

```bash
docker-compose up -d redis jaeger
```

Проверь что запустились:
```bash
docker ps
redis-cli ping  # Должно вернуть PONG
```

## 9. Запуск vLLM

**В отдельном терминале:**

```bash
source venv/bin/activate

vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --quantization awq \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

Ожидай пока модель загрузится (~2-3 минуты).

## 10. Запуск сервисов

**Каждый в отдельном терминале:**

### Terminal 1: ASR Gateway
```bash
source venv/bin/activate
./venv/bin/python src/asr_gateway/main.py
```

### Terminal 2: TTS Gateway
```bash
source venv/bin/activate
./venv/bin/python src/tts_gateway/main.py
```

### Terminal 3: Policy Engine
```bash
source venv/bin/activate
./venv/bin/python src/policy_engine/main.py
```

## 11. Health Check

**В новом терминале:**

```bash
source venv/bin/activate
./venv/bin/python scripts/health_check.py
```

Ожидаемый вывод:
```
✅ Environment: Python 3.12.3, CUDA available, Google Credentials found
✅ GPU: RTX 5090 (32 GB)
✅ Redis: running
✅ Services:
   ASR Gateway.............. ✅ healthy
   LLM Service.............. ✅ healthy
   TTS Gateway.............. ✅ healthy
   Policy Engine............ ✅ healthy
```

## 12. Тестирование диалога

```bash
# Отправить тестовое сообщение в Policy Engine
curl -X POST http://localhost:8003/dialog \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-1",
    "user_message": "Здравствуйте"
  }' | jq
```

Ожидаемый ответ:
```json
{
  "session_id": "test-session-1",
  "agent_message": "Добрый день! Меня зовут администратор медицинского центра МРТ 1.5Т. Чем могу вам помочь?",
  "current_state": "greeting",
  "slots": {},
  "is_complete": false
}
```

## 13. Мониторинг

- **Jaeger UI**: http://localhost:16686
- **Health endpoints**:
  - ASR: http://localhost:8001/health
  - TTS: http://localhost:8002/health
  - Policy: http://localhost:8003/health
  - LLM: http://localhost:8000/v1/models

## Troubleshooting

### vLLM не запускается

Проверь VRAM:
```bash
nvidia-smi
```

Должно быть минимум 12 GB свободно.

### Service not running

Проверь логи в stdout сервиса. Если видишь ошибку, читай раздел "Частые ошибки" в README.md.

### Google Sheets permission denied

1. Открой Google Sheets
2. Нажми "Share"
3. Добавь email из `credentials/google_credentials.json` (`client_email`)
4. Дай права "Editor"

## Следующие шаги

- Прочитай [README.md](README.md) для overview
- Прочитай [OVERVIEW.md](OVERVIEW.md) для архитектуры
- Изучи документацию каждого сервиса в `src/*/README.md`

Удачи! 🚀

