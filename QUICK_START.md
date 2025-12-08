# 🚀 Инструкция по запуску и тестированию Sales Agent

## 📋 Команды для запуска сервисов

Запустите каждый сервис в **отдельном терминале**. Порядок запуска важен!

### Терминал 1: Redis
```bash
cd /path/to/ai-agent-TTS
docker-compose up -d redis
```

**Проверка:** `docker ps | grep redis` должен показать запущенный контейнер

---

### Терминал 2: vLLM сервер (LLM модель)
```bash
cd /path/to/ai-agent-TTS
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3-16B-A3B-abliterated-AWQ \
  --host 0.0.0.0 --port 8000 \
  --quantization awq \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75
```

**Ожидайте:** Сообщение `Application startup complete` (загрузка модели займет 1-2 минуты)

**Проверка:** `curl http://localhost:8000/v1/models` должен вернуть JSON с информацией о модели

---

### Терминал 3: ASR Gateway (распознавание речи)
```bash
cd /path/to/ai-agent-TTS
uv run python src/asr_gateway/main.py
```

**Ожидайте:** Сообщение `ASR Gateway started successfully`

**Проверка:** `curl http://localhost:8001/health` должен вернуть `{"status": "healthy"}`

---

### Терминал 4: TTS Gateway (синтез речи)
```bash
cd /path/to/ai-agent-TTS
uv run python src/tts_gateway/main.py
```

**Ожидайте:** Сообщение `TTS Gateway started successfully`

**Проверка:** `curl http://localhost:8002/health` должен вернуть `{"status": "healthy"}`

---

### Терминал 5: Policy Engine (оркестратор диалога)
```bash
cd /path/to/ai-agent-TTS
uv run python src/policy_engine/main.py
```

**Ожидайте:** Сообщение `Policy Engine started successfully`

**Проверка:** `curl http://localhost:8003/health` должен вернуть `{"status": "healthy"}`

---

## ✅ Проверка что все сервисы запущены

Выполните в любом терминале:
```bash
# Redis
docker ps | grep redis

# vLLM
curl http://localhost:8000/v1/models

# ASR Gateway
curl http://localhost:8001/health

# TTS Gateway
curl http://localhost:8002/health

# Policy Engine
curl http://localhost:8003/health
```

Все команды должны вернуть успешный ответ.

---

## 🧪 Запуск тестирования

После того как все сервисы запущены и отвечают, запустите симуляцию диалога:

```bash
cd /path/to/ai-agent-TTS
uv run python scripts/test_dialog_performance.py --scenario-name basic_success
```

## 📊 Что измеряет симулятор

- ⏱️ **Латентность ответа** - время от запроса клиента до ответа агента (цель: 250-600 мс)
- 🔄 **Переходы FSM** - корректность переходов между состояниями диалога
- 📝 **Заполнение слотов** - сколько информации собрано о клиенте
- 📊 **E2E метрики** - общее время диалога и средняя латентность

## 🎭 Доступные сценарии

- `basic_success` - базовый успешный сценарий (клиент проходит все этапы)
- `with_objections` - с возражениями клиента
- `quick_booking` - быстрая запись (клиент знает что хочет)
- `with_clarifications` - с уточнениями

## 📝 Примеры команд для тестирования

```bash
# Полный диалог с базовым сценарием
uv run python scripts/test_dialog_performance.py --scenario-name basic_success

# Короткий тест (10 ходов)
uv run python scripts/test_dialog_performance.py --scenario-name quick_booking --max-turns 10

# Тест с возражениями
uv run python scripts/test_dialog_performance.py --scenario-name with_objections
```

## 📁 Логи сервисов

Логи выводятся в терминалы где запущены сервисы. Для отладки можно перенаправить вывод:

```bash
# vLLM
uv run python -m vllm.entrypoints.openai.api_server ... > /tmp/vllm.log 2>&1

# Policy Engine
uv run python src/policy_engine/main.py > /tmp/policy_engine.log 2>&1

# ASR Gateway
uv run python src/asr_gateway/main.py > /tmp/asr_gateway.log 2>&1

# TTS Gateway
uv run python src/tts_gateway/main.py > /tmp/tts_gateway.log 2>&1
```

## 🛑 Остановка сервисов

Для остановки нажмите `Ctrl+C` в каждом терминале с сервисом, или:

```bash
# Остановить все Python сервисы
pkill -f "vllm"
pkill -f "policy_engine"
pkill -f "asr_gateway"
pkill -f "tts_gateway"

# Остановить Redis
docker-compose stop redis
```

