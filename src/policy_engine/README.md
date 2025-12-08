# Policy Engine

Оркестратор диалога с **LangGraph FSM** для записи на МРТ (30 этапов из `script_evaluation_type_A.md`).

## Возможности

- ✅ FSM с 30 этапами диалога
- ✅ Извлечение слотов через LLM (structured output)
- ✅ Валидация слотов (телефон, имя, дата/время)
- ✅ Хранение состояния сессий в Redis
- ✅ Интеграция с LLM (vLLM + Qwen2.5-14B-AWQ)
- ✅ Асинхронная запись в Google Sheets (через notifier)
- ✅ OpenTelemetry трейсинг

## Архитектура

```
Client → Policy Engine
    ├─ FSM (управляет состояниями)
    ├─ LLM (извлекает слоты + генерирует ответы)
    ├─ Redis (хранит состояния сессий)
    └─ Notifier (записывает в Google Sheets)
```

## Требования

- vLLM сервер должен быть запущен (http://localhost:8000)
- Redis (docker-compose up redis)

## Запуск

```bash
# Убедись что Redis запущен
docker-compose up -d redis

# Запусти Policy Engine через uv
uv run python src/policy_engine/main.py
```

## API

### POST `/dialog`

Обрабатывает сообщение в диалоге:

```bash
curl -X POST http://localhost:8003/dialog \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-123",
    "user_message": "Добрый день, хочу записаться на МРТ"
  }'
```

Ответ:

```json
{
  "session_id": "test-session-123",
  "agent_message": "Добрый день! Меня зовут администратор медицинского центра МРТ 1.5Т. Чем могу вам помочь?",
  "current_state": "greeting",
  "slots": {},
  "is_complete": false
}
```

### GET `/session/{session_id}`

Получает текущее состояние сессии:

```bash
curl http://localhost:8003/session/test-session-123
```

### DELETE `/session/{session_id}`

Удаляет сессию:

```bash
curl -X DELETE http://localhost:8003/session/test-session-123
```

## FSM States (30 этапов)

Основные этапы диалога:

1. **START** → **GREETING**: Приветствие, название центра, представление
2. **ASK_CLIENT_NAME**: Спросить имя клиента
3. **ASK_SYMPTOMS**: Узнать жалобы/симптомы
4. **ASK_SYMPTOMS_DURATION**: Длительность симптомов
5. **ASK_PAIN_CHARACTER**: Характер боли
6. **ASK_VISITED_DOCTOR**: Был ли у врача
7. **ASK_STUDY_REQUEST**: Что именно хочет клиент
8. **RECOMMEND_STUDY**: Рекомендация комплекса + аргументы
9. **ANNOUNCE_PRICE**: Стоимость исследования
10. **ASK_STUDY_DECISION**: Выбор клиента
11. **OFFER_VIDEO_CONCLUSION**: Предложение видеозаключения + описание
12. **ANNOUNCE_MEDIA_PRICE**: Стоимость носителя
13. **ASK_APPOINTMENT_DATE**: Дата записи
14. **OFFER_APPOINTMENT_TIMES**: Варианты времени (не открытый вопрос!)
15. **CONFIRM_TIME**: Уточнение удобства времени
16. **ASK_PHONE**: Телефон для связи
17. **ASK_AGE_WEIGHT**: Возраст и вес
18. **CHECK_CONTRAINDICATIONS**: Противопоказания
19. **CHECK_DISCOUNTS**: Скидки и льготы
20. **REMIND_DOCUMENTS**: Документы и подготовка
21. **PROVIDE_ADDRESS**: Адрес центра
22. **PROVIDE_CONTACTS**: Телефон центра
23. **CONFIRM_APPOINTMENT**: Итоговое подтверждение (дата, время, цена)
24. **FAREWELL**: Прощание
25. **END**: Завершение

## Slot Extraction

LLM извлекает слоты из сообщений клиента:

**Пример:**

Клиент: *"Меня зовут Иван, телефон +79991234567, у меня болит голова"*

Извлечённые слоты:
```json
{
  "client_name": "Иван",
  "client_phone": "+79991234567",
  "symptoms": "болит голова"
}
```

## Интеграция с Google Sheets

Когда диалог завершён (`is_complete=true`), слоты автоматически записываются в Google Sheets (Лист4):

- timestamp
- client_name
- client_phone
- symptoms
- study_type
- appointment_date
- appointment_time
- study_price
- total_price
- status

**Асинхронная запись**: не блокирует диалог.

## Приёмка

1. **Запуск без ошибок**:
   ```bash
   uv run python src/policy_engine/main.py
   ```

2. **Health check**:
   ```bash
   curl http://localhost:8003/health
   ```

3. **Диалог работает**:
   ```bash
   curl -X POST http://localhost:8003/dialog \
     -H "Content-Type: application/json" \
     -d '{"session_id": "test-1", "user_message": "Здравствуйте"}' | jq
   ```

4. **Слоты извлекаются**:
   Отправь несколько сообщений, проверь что `slots` заполняются

5. **FSM переходит между состояниями**:
   Проверь поле `current_state` в ответе

## Метрики

- **Slot extraction latency**: 100-300 мс
- **Response generation latency**: 40-150 мс
- **Total turn latency**: 150-450 мс

## Логи

```json
{
  "timestamp": "2025-10-24T19:45:00.123Z",
  "level": "INFO",
  "service": "policy_engine",
  "message": "FSM: greeting -> ask_client_name",
  "context": {"session_id": "test-1"}
}
```

## Roadmap

- [ ] Интеграция с LangGraph для визуализации FSM
- [ ] Поддержка multi-turn контекста (история > 10 сообщений)
- [ ] A/B тестирование промптов
- [ ] Автоматическое переобучение на реальных диалогах

