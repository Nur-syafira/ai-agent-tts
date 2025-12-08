# Low-Latency Voice AI Agent

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![Type checking](https://img.shields.io/badge/type%20checking-pyright-yellow.svg)](https://github.com/microsoft/pyright)

> Модульная платформа для создания голосовых AI-агентов с минимальной латентностью (250-600 мс E2E)

## Проблема

Создание production-ready голосовых AI-агентов требует решения нескольких сложных задач одновременно:
- **Низкая латентность** (рот→ухо < 600 мс) для естественного диалога
- **Потоковая обработка** аудио без буферизации
- **Управление диалогом** через FSM с извлечением структурированных данных
- **Масштабируемость** через микросервисную архитектуру
- **Мониторинг** E2E метрик для отладки

Большинство существующих решений либо слишком медленные (>1s латентность), либо монолитные, либо не предоставляют готовую архитектуру для production.

## Решение

Модульная платформа на базе микросервисов с оптимизированными компонентами:
- **ASR Gateway**: потоковое распознавание речи (RealtimeSTT + faster-whisper, 80-150 мс)
- **TTS Gateway**: синтез речи с кэшированием (F5-TTS, 50-150 мс)
- **LLM Service**: инференс через vLLM с AWQ квантизацией (40-150 мс)
- **Policy Engine**: управление диалогом через LangGraph FSM + Pydantic slots
- **Мониторинг**: OpenTelemetry для E2E трейсинга

Архитектура позволяет запускать все компоненты локально на одной GPU (12+ GB VRAM) или масштабировать горизонтально.

## 🚀 Возможности

- ✅ **Потоковое ASR**: RealtimeSTT + faster-whisper large-v3-turbo (80-150 мс)
- ✅ **Silero VAD**: Endpointing + barge-in detection (100-200 мс)
- ✅ **LLM**: Qwen3-16B-A3B-abliterated-AWQ через vLLM (40-150 мс, MoE архитектура)
- ✅ **TTS**: F5-TTS (50-150 мс, русский) + пререндер (<10 мс)
- ✅ **FSM**: Гибкая система управления диалогом (настраивается под любой домен)
- ✅ **Google Sheets**: Append-only запись в Лист4
- ✅ **FreeSWITCH Bridge**: WebSocket интеграция с mod_audio_fork
- ✅ **CUDA-only**: Guard-проверки GPU при старте
- ✅ **OpenTelemetry**: E2E латентность мониторинг

## 📊 Целевые метрики

| Компонент | Латентность | VRAM |
|-----------|-------------|------|
| ASR partial | 80-150 мс | ~3 GB |
| LLM inference | 40-150 мс | ~6 GB (MoE модель) |
| TTS first-audio | 50-120 мс | ~1 GB |
| **E2E (рот→ухо)** | **250-600 мс** | **~10 GB** |

## 🛠️ Технологический стек

- **ASR**: RealtimeSTT + faster-whisper large-v3-turbo
- **VAD**: Silero VAD v5
- **LLM**: Qwen3-16B-A3B-abliterated-AWQ (vLLM, MoE архитектура)
- **TTS**: F5-TTS (русский)
- **Policy**: LangGraph FSM + Pydantic slots
- **Storage**: Redis (сессии)
- **Sheets**: gspread-asyncio
- **Monitoring**: OpenTelemetry + Jaeger

## 📋 Требования

- **OS**: Linux (Ubuntu 22.04+)
- **Python**: 3.12
- **Менеджер пакетов**: [uv](https://github.com/astral-sh/uv) (рекомендуется) или pip
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

## Быстрый старт

```bash
# 1. Клонировать и установить
git clone https://github.com/YOUR_USERNAME/ai-agent-TTS.git
cd ai-agent-TTS
uv sync

# 2. Скачать модели
uv run python scripts/download_models.py

# 3. Настроить окружение
cp .env.example .env
# Отредактировать .env (укажите свои пути)

# 4. Запустить сервисы
docker-compose up -d redis jaeger
uv run python scripts/start_services.sh

# 5. Проверить здоровье
uv run python scripts/health_check.py
```

**Минимальный пример использования:**

```python
from src.policy_engine.main import PolicyEngine
from src.policy_engine.slots import DialogSlots

# Инициализация (автоматически подключается к ASR/TTS/LLM)
engine = PolicyEngine()

# Обработка сообщения пользователя
response = await engine.process_message(
    session_id="test-123",
    user_message="Здравствуйте, хочу записаться на МРТ"
)

print(response.agent_message)  # Ответ агента
print(response.slots)  # Извлеченные данные
```

Подробнее см. [QUICK_START.md](QUICK_START.md) и [examples/](examples/).

## Примеры

См. папку `examples/` — готовый рабочий код:

- `examples/basic_dialog.py` — базовый диалог
- `examples/custom_fsm.py` — кастомизация FSM
- `examples/custom_prompts.py` — настройка промптов

## Расширяемость

Платформа легко расширяется через конфигурацию:

```python
# Кастомный FSM
from src.policy_engine.fsm import DialogFSM, DialogState

class MyFSM(DialogFSM):
    def _build_transitions(self):
        # Ваша логика переходов
        return [...]
```

```yaml
# config.yaml
llm:
  model_name: "your-model"
  temperature: 0.7
```

## Документация

- [OVERVIEW.md](OVERVIEW.md) — архитектура и потоки данных
- [QUICK_START.md](QUICK_START.md) — подробная инструкция по запуску
- [src/asr_gateway/README.md](src/asr_gateway/README.md) — ASR сервис
- [src/llm_service/README.md](src/llm_service/README.md) — LLM сервис
- [src/tts_gateway/README.md](src/tts_gateway/README.md) — TTS сервис
- [src/policy_engine/README.md](src/policy_engine/README.md) — Policy Engine
- [src/notifier/README.md](src/notifier/README.md) — Google Sheets интеграция
- [src/freeswitch_bridge/README.md](src/freeswitch_bridge/README.md) — FreeSWITCH интеграция

## 🧪 Тестирование

```bash
# Запустить все тесты
uv run pytest tests/ -v

# Тесты с покрытием
uv run pytest tests/ --cov=src --cov-report=html

# Симуляция полного диалога
uv run python scripts/simulate_dialog.py

# Симуляция с конкретным сценарием
uv run python scripts/simulate_dialog.py --scenario scripts/dialog_scenarios.yaml
```

## 📊 Мониторинг

### Prometheus метрики

Каждый сервис экспортирует Prometheus метрики на endpoint `/metrics`:

- `{service}_requests_total` - общее количество запросов
- `{service}_request_latency_seconds` - латентность запросов
- `{service}_active_connections` - количество активных соединений
- `{service}_errors_total` - количество ошибок

Пример настройки Prometheus:

```yaml
scrape_configs:
  - job_name: 'sales-agent'
    static_configs:
      - targets: ['localhost:8001', 'localhost:8002', 'localhost:8003']
```

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

## 🐳 Docker

Проект поддерживает контейнеризацию через Docker и docker-compose.

### Сборка образа

```bash
# Сборка Docker образа
docker build -t sales-agent:latest .

# Или через docker-compose
docker-compose build
```

### Запуск через docker-compose

```bash
# Запуск всех сервисов (Redis, Jaeger, ASR Gateway, TTS Gateway, Policy Engine)
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

**Примечание**: Для работы с GPU в Docker требуется настройка NVIDIA Container Toolkit.

## 🔄 CI/CD

Проект использует GitHub Actions для автоматической проверки кода и сборки образов.

### Workflow включает:

- **Линтинг**: ruff (форматирование и проверка кода), pyright (проверка типов)
- **Тестирование**: pytest с покрытием кода
- **Security scan**: bandit (безопасность кода), pip-audit (уязвимости зависимостей), Trivy (сканирование Docker образов)
- **Сборка Docker**: автоматическая сборка образов при push в main/master

### Локальный запуск проверок

```bash
# Установить dev-зависимости
uv sync --group dev

# Запустить линтинг
uv run ruff check .
uv run ruff format --check .
uv run pyright src/

# Запустить тесты
uv run pytest tests/ -v --cov=src

# Проверка безопасности
uv run bandit -r src/
uv run pip-audit
```

### Pre-commit hooks

```bash
# Установить pre-commit hooks
uv run pre-commit install

# Запустить проверки вручную
uv run pre-commit run --all-files
```

## 🔒 Безопасность

- ✅ Credentials в `.gitignore` (никогда не коммитятся)
- ✅ Service Account (не personal account) для Google Sheets
- ✅ `.env` не коммитится (есть `.env.example` как шаблон)
- ✅ Минимальные права доступа
- ✅ Security scan в CI/CD (bandit, pip-audit, Trivy)
- ✅ Pre-commit hooks для проверки секретов (gitleaks)
- ✅ Контейнеры запускаются от непривилегированного пользователя

### Проверка безопасности

```bash
# Локальная проверка безопасности кода
uv run bandit -r src/

# Проверка уязвимостей зависимостей
uv run pip-audit

# Проверка на hardcoded секреты
pre-commit run gitleaks --all-files
```

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
uv run python src/asr_gateway/main.py  # Смотреть stdout
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

## Лицензия

Apache 2.0 — см. [LICENSE](LICENSE)

## Цитирование

```bibtex
@software{ai-agent-tts2025,
  title = {Low-Latency Voice AI Agent},
  author = {Mordvinov, Aleksandr},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/ai-agent-TTS}
}
```

## 💎 Premium Features

Ищете готовые решения?

- 🎯 **Domain-Specific Prompts** — Оптимизированные промпты для медицинских центров, продаж, поддержки
- 🚀 **Advanced FSM Templates** — Готовые диалоговые сценарии
- ⚡ **Performance Optimization Pack** — Продвинутые техники оптимизации

*Premium функции доступны отдельно. [Узнать больше →](mailto:premium@yourdomain.com)*

## 🤝 Консалтинг и интеграции

Нужна кастомизация или интеграция?

- Разработка кастомных FSM под ваш домен
- Оптимизация промптов под вашу задачу
- Поддержка production deployment
- Обучение команды

*[Связаться с нами →](mailto:consulting@yourdomain.com)*

## Roadmap

- [ ] True streaming TTS (не чанки после синтеза)
- [ ] A/B testing промптов
- [ ] Multi-turn context (>10 сообщений)
- [ ] WebRTC для web-демо
- [ ] Горизонтальное масштабирование (load balancer)
- [ ] Multi-tenant support
- [ ] Dashboard (Grafana)

## 👤 Author

**Aleksandr Mordvinov**

## 🙏 Acknowledgments

- **Qwen Team** — Qwen2.5 LLM
- **Systran** — faster-whisper
- **Silero Team** — Silero VAD
- **SWivid** — F5-TTS
- **Misha24-10** — F5-TTS Russian model
- **vLLM Team** — vLLM inference engine

---

Made with ❤️ for low-latency voice AI

