# Tests

Тесты для Sales Agent MVP.

## Запуск тестов

```bash
# Все тесты
./venv/bin/pytest tests/ -v

# С покрытием
./venv/bin/pytest tests/ --cov=src --cov-report=html

# Отдельный тест
./venv/bin/pytest tests/test_slots.py -v

# Только unit тесты
./venv/bin/pytest tests/ -v -m "not integration"
```

## Структура

- `conftest.py` - fixtures и конфигурация pytest
- `test_config_loader.py` - тесты для загрузки конфигурации
- `test_slots.py` - тесты для Pydantic моделей слотов
- `test_fsm.py` - тесты для FSM (Finite State Machine)
- `test_health.py` - тесты для health checks

## TODO

- [ ] Integration tests (тестирование с реальными сервисами)
- [ ] E2E latency tests
- [ ] Load tests
- [ ] Mock tests для Google Sheets API
- [ ] WebSocket tests для ASR Gateway

