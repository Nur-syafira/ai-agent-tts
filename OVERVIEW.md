# Sales Agent MVP - Architecture Overview

Детальное описание архитектуры, потоков данных и дизайн-решений.

## 📐 Высокоуровневая архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                          Телефония (SIP)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    FreeSWITCH
                  (mod_audio_fork)
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
  Входящий           Исходящий          WebSocket
   Audio              Audio              Bridge
(от клиента)      (к клиенту)        (двунаправленный)
      │                  ▲                  │
      │                  │                  │
      ▼                  │                  ▼
┌─────────────┐          │          ┌──────────────┐
│ ASR Gateway │          │          │ TTS Gateway  │
│  (RealtimeSTT)│        │          │   (Piper)    │
│  + Silero VAD│          │          │ + Kokoro-82M │
└──────┬──────┘          │          └──────▲───────┘
       │                  │                 │
       │ Transcript       │                 │ Text
       │                  │                 │
       ▼                  │                 │
┌────────────────────────────────────────────────┐
│            Policy Engine (FSM)                  │
│  ┌──────────────────────────────────────────┐  │
│  │  LangGraph FSM (30 states)               │  │
│  │    ↓                                      │  │
│  │  Slot Extraction (LLM structured output) │  │
│  │    ↓                                      │  │
│  │  State Transition Logic                  │  │
│  │    ↓                                      │  │
│  │  Response Generation (LLM)               │  │
│  └──────────────────────────────────────────┘  │
│                     ↓                           │
│              Redis (Sessions)                   │
└──────────────────┬─────────────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ LLM Service    │
          │ (vLLM Server)  │
          │ Qwen2.5-14B-AWQ│
          └────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │Google Sheets   │
          │   Notifier     │
          │   (Лист4)      │
          └────────────────┘
```

## 🔄 Поток данных E2E

### 1. Входящий звонок

```
Клиент говорит → SIP → FreeSWITCH → WebSocket → ASR Gateway
```

**Формат аудио:**
- PCM 16-bit mono
- 16 kHz sample rate
- 160 мс чанки (2560 сэмплов)

### 2. Распознавание речи

```
ASR Gateway:
  Audio chunk (160 ms)
    ↓
  Silero VAD (30-60 ms window)
    ↓
  Speech detected? → faster-whisper inference
    ↓
  Partial transcript (каждые 120 мс)
    ↓
  Endpointing (500 мс тишины) → Final transcript
```

**Латентность ASR:**
- Partial: 80-150 мс
- Final (после endpointing): 150-250 мс

### 3. Обработка диалога

```
Policy Engine:
  User message
    ↓
  LLM Slot Extraction (structured JSON)
    ↓
  Update DialogSlots
    ↓
  FSM State Transition
    ↓
  LLM Response Generation
    ↓
  Agent message
```

**Латентность Policy:**
- Slot extraction: 100-300 мс
- Response generation: 40-150 мс
- **Total**: 150-450 мс

### 4. Синтез речи

```
TTS Gateway:
  Agent text
    ↓
  Check prerender cache (Redis/file)
    ↓
  Cache HIT? → Return immediately (<10 ms)
    ↓
  Cache MISS? → Piper synthesis (80-150 ms)
    ↓
  PCM audio chunks (200 ms)
    ↓
  Stream to FreeSWITCH
```

**Латентность TTS:**
- Cached: <10 мс
- Uncached: 80-150 мс

### 5. Исходящий audio

```
TTS Gateway → WebSocket → FreeSWITCH → SIP → Клиент слышит
```

**Джиттер буфер:** 200-300 мс

### 6. Сохранение данных

```
Dialog complete?
  ↓
DialogSlots → Google Sheets Notifier (async)
  ↓
Append to Лист4 (не блокирует диалог)
```

## 🧩 Компоненты

### ASR Gateway (`src/asr_gateway/`)

**Роль:** Потоковое распознавание речи

**Технологии:**
- RealtimeSTT (faster-whisper wrapper)
- faster-whisper large-v3-turbo (INT8)
- Silero VAD v5

**Endpoints:**
- `WS /ws/transcribe` — WebSocket для стриминга
- `GET /health` — Health check
- `GET /ready` — Readiness check

**Оптимизации:**
- beam_size=1 (минимальная латентность)
- condition_on_previous_text=false
- Partial transcripts каждые 120 мс
- CUDA-only (guard check при старте)

### LLM Service (`src/llm_service/`)

**Роль:** Инференс LLM через vLLM

**Технологии:**
- vLLM 0.6.7
- Qwen2.5-14B-Instruct-AWQ (INT4)
- Flash Attention 2
- PagedAttention

**Endpoints:**
- `POST /v1/chat/completions` — OpenAI-compatible
- `GET /v1/models` — Список моделей

**Оптимизации:**
- AWQ квантизация (INT4)
- Chunked prefill
- Prefix caching (для system prompt)
- Structured output (JSON mode)

### TTS Gateway (`src/tts_gateway/`)

**Роль:** Синтез речи

**Технологии:**
- Piper TTS (основной)
- Kokoro-82M (опционально, в разработке)
- Redis + file cache

**Endpoints:**
- `POST /synthesize` — Синтез текста в аудио
- `GET /health` — Health check

**Оптимизации:**
- Пререндер 20-30 частых фраз
- Redis кэш (TTL 1 час)
- Streaming output (чанки 200 мс)

### Policy Engine (`src/policy_engine/`)

**Роль:** Оркестрация диалога

**Технологии:**
- LangGraph FSM
- Pydantic slots
- Redis (session storage)

**Endpoints:**
- `POST /dialog` — Обработка сообщения
- `GET /session/{id}` — Получить состояние сессии
- `DELETE /session/{id}` — Удалить сессию

**FSM States (30):**
1. GREETING
2. ASK_CLIENT_NAME
3. ASK_SYMPTOMS
4. ASK_SYMPTOMS_DURATION
5. ASK_PAIN_CHARACTER
6. ASK_VISITED_DOCTOR
7. ASK_STUDY_REQUEST
8. RECOMMEND_STUDY
9. ANNOUNCE_PRICE
10. ASK_STUDY_DECISION
11. OFFER_VIDEO_CONCLUSION
12. ANNOUNCE_MEDIA_PRICE
13. ASK_APPOINTMENT_DATE
14. OFFER_APPOINTMENT_TIMES
15. CONFIRM_TIME
16. ASK_PHONE
17. ASK_AGE_WEIGHT
18. CHECK_CONTRAINDICATIONS
19. CHECK_DISCOUNTS
20. REMIND_DOCUMENTS
21. PROVIDE_ADDRESS
22. PROVIDE_CONTACTS
23. CONFIRM_APPOINTMENT
24. FAREWELL
25. END

### Google Sheets Notifier (`src/notifier/`)

**Роль:** Запись результатов в Google Sheets

**Технологии:**
- gspread-asyncio
- Google Service Account
- tenacity (retry)

**Формат Лист4:**
- Timestamp
- Имя клиента
- Телефон
- Возраст
- Вес
- Симптомы
- Тип исследования
- Дата/время записи
- Стоимость
- Статус

**Режим:** Append-only (не очищается между запусками)

### FreeSWITCH Bridge (`src/freeswitch_bridge/`)

**Роль:** Интеграция с телефонией

**Статус:** ⚠️ В разработке

**Технологии:**
- FreeSWITCH mod_audio_fork
- WebSocket dual-channel audio
- Barge-in detection (двухканальный VAD)

## 🎯 Оптимизации производительности

### 1. CUDA оптимизации

```bash
# GPU persistence mode
sudo nvidia-smi -pm 1

# Проверка compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Эффект:** -10-20 мс на инициализацию моделей

### 2. CPU оптимизации

```bash
# CPU governor = performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Эффект:** -5-10 мс на обработку аудио

### 3. Network оптимизации

```python
# uvloop вместо asyncio
import uvloop
uvloop.install()
```

**Эффект:** -20-30% latency на I/O операциях

### 4. LLM оптимизации

- **AWQ**: -30% latency vs FP16
- **Flash Attention 2**: -20% latency vs стандартного attention
- **Prefix caching**: -50% latency на повторяющихся промптах

### 5. TTS оптимизации

- **Prerender**: -140 мс на частых фразах
- **Redis cache**: -80 мс на кэшированных фразах

## 📊 Расчёт VRAM

| Компонент | VRAM |
|-----------|------|
| faster-whisper large-v3-turbo (INT8) | 3 GB |
| Qwen2.5-14B-AWQ (INT4) | 8 GB |
| Kokoro-82M (ONNX) | 0.5 GB |
| Buffers & cache | 0.5 GB |
| **Total** | **12 GB** |

**RTX 5090 (32 GB)** → **остаётся 20 GB** запаса

## 🔐 Безопасность

### Credentials

```
credentials/
  └── google_credentials.json  # Service Account (не коммитится)
```

### Secrets management

```bash
# .env файл
GOOGLE_CREDENTIALS_PATH=/path/to/credentials.json
GOOGLE_SHEET_ID=1Fh7K3shc...
REDIS_URL=redis://localhost:6379
```

### Network

- Все сервисы на `localhost` (не exposed наружу)
- FreeSWITCH → единственная точка входа
- HTTPS/TLS для production

## 📈 Масштабирование

### Горизонтальное (будущее)

```
Load Balancer
  ├─ ASR Gateway #1
  ├─ ASR Gateway #2
  ├─ ...
  └─ ASR Gateway #N
```

**Shared:**
- Redis (session storage)
- vLLM (shared KV cache)

### Вертикальное (текущее)

- Один мощный сервер (RTX 5090 + Ryzen 9950X3D)
- Max concurrent sessions: 10-20

## 🧪 Тестирование

### Unit tests

```bash
pytest tests/test_asr.py -v
pytest tests/test_policy.py -v
```

### Integration tests

```bash
pytest tests/test_integration.py -v
```

### E2E latency test

```python
# Симулировать полный диалог
# Измерить E2E латентность
# Target: 280-640 мс
```

## 📝 Логирование

### Формат

```json
{
  "timestamp": "2025-10-24T19:30:00.123Z",
  "level": "INFO",
  "service": "policy_engine",
  "message": "FSM transition",
  "context": {
    "from_state": "greeting",
    "to_state": "ask_client_name",
    "session_id": "abc-123"
  }
}
```

### Уровни

- **DEBUG**: Детальные логи (FSM transitions, slot updates)
- **INFO**: Важные события (session start/end, API calls)
- **WARNING**: Неожиданные ситуации (fallback to Piper, retry)
- **ERROR**: Ошибки (API failures, exceptions)

## 🎓 Best Practices

1. **Никаких fallback/заглушек** — только реальные модули
2. **Guard checks** — fail fast при отсутствии GPU/Redis
3. **Структурированные логи** — JSON, не print
4. **Валидация конфигов** — Pydantic при старте
5. **Append-only** — Google Sheets не очищается
6. **Idempotency** — повторный запуск не создаёт дубликатов
7. **Timeouts & retries** — tenacity с exp backoff
8. **Health checks** — `/health` и `/ready` endpoints

## 🚀 Roadmap

### Phase 1 (MVP) — ✅ Current
- [x] ASR Gateway
- [x] LLM Service
- [x] TTS Gateway
- [x] Policy Engine (FSM)
- [x] Google Sheets Notifier
- [ ] FreeSWITCH Bridge

### Phase 2 (Production)
- [ ] True Kokoro-82M integration
- [ ] True streaming TTS (не чанки после синтеза)
- [ ] A/B testing промптов
- [ ] Multi-turn context (>10 сообщений)
- [ ] WebRTC для web-демо

### Phase 3 (Scale)
- [ ] Horizontal scaling (load balancer)
- [ ] Multi-tenant support
- [ ] Dashboard (Grafana)
- [ ] Call recording & playback
- [ ] Analytics & reporting

---

**Документация актуальна на:** 2025-10-24  
**Версия:** 0.1.0  
**Автор:** Aleksandr Mordvinov

