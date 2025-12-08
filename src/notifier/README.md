# Notifier

Сервис интеграции с **Google Sheets** для записи результатов диалога в **Лист4**.

## Возможности

- ✅ Async запись в Google Sheets через gspread-asyncio
- ✅ **Append-only** режим (не очищает лист между запусками)
- ✅ Retry с exponential backoff
- ✅ Валидация данных перед записью
- ✅ Не блокирует диалог (async)

## Требования

- Google Service Account с доступом к таблице
- Credentials в `credentials/google_credentials.json`

## Конфигурация

Редактируйте `config.yaml`:

```yaml
google_sheets:
  credentials_path: "credentials/google_credentials.json"
  spreadsheet_id: "YOUR_GOOGLE_SHEET_ID"  # Замените на ID вашей таблицы
  worksheet_name: "Лист4"
  append_only: true  # ВАЖНО: не очищать лист!

retry:
  max_attempts: 3
  initial_delay: 1.0
  backoff_factor: 2.0
```

## Использование

### Из Policy Engine

```python
from src.notifier.sheets_client import GoogleSheetsClient
from src.notifier.formatter import format_slots_for_sheets

# Инициализация
client = GoogleSheetsClient(
    credentials_path="credentials/google_credentials.json",
    spreadsheet_id="YOUR_GOOGLE_SHEET_ID",
    worksheet_name="Лист4",
)

# Запись данных
row_data = format_slots_for_sheets(dialog_slots)
await client.append_row(row_data)
```

### Standalone скрипт

```python
import asyncio
from src.notifier.sheets_client import GoogleSheetsClient

async def main():
    client = GoogleSheetsClient(
        credentials_path="credentials/google_credentials.json",
        spreadsheet_id="YOUR_SHEET_ID",
        worksheet_name="Лист4",
    )
    
    # Добавить строку
    await client.append_row({
        "client_name": "Иван Иванов",
        "client_phone": "+79991234567",
        "symptoms": "Боль в голове",
        "study_type": "МРТ головного мозга",
        "appointment_date": "2025-10-25",
        "appointment_time": "15:00",
        "study_price": 4500,
        "total_price": 4800,
        "status": "записан",
    })
    
    print("✅ Данные записаны!")

if __name__ == "__main__":
    asyncio.run(main())
```

## Формат данных в Лист4

| Колонка | Тип | Описание |
|---------|-----|----------|
| Timestamp | datetime | Время записи (UTC) |
| Имя клиента | string | ФИО клиента |
| Телефон | string | +7XXXXXXXXXX |
| Возраст | int | Возраст клиента |
| Вес (кг) | float | Вес клиента |
| Симптомы | string | Жалобы клиента |
| Тип исследования | string | МРТ головного мозга/комплекс |
| Дата записи | string | YYYY-MM-DD |
| Время записи | string | HH:MM |
| Стоимость исследования | float | Цена исследования |
| Формат заключения | string | бумажное/видеозаключение |
| Стоимость носителя | float | Цена диска/флешки |
| Итоговая стоимость | float | Сумма |
| Статус | string | записан/отменён/выполнен |

## Append-Only режим

**ВАЖНО**: По требованиям проекта, Google Sheet работает в **append-only** режиме. Это значит:

- ✅ Новые записи **добавляются** в конец листа
- ❌ Лист **НЕ очищается** между запусками
- ✅ Все данные сохраняются

Метод `clear_sheet()` доступен только для тестов и не должен использоваться в проде.

## Retry политика

При ошибках API Google Sheets, клиент автоматически повторяет запрос:

- Попытки: 3
- Задержка: 1s → 2s → 4s (exponential backoff)
- Макс. задержка: 10s

## Приёмка

1. **Проверка credentials**:
   ```bash
   ls -la credentials/google_credentials.json
   ```

2. **Тест записи**:
   ```python
   # Создать test_notifier.py
   import asyncio
   from src.notifier.sheets_client import GoogleSheetsClient
   
   async def test():
       client = GoogleSheetsClient(
           credentials_path="credentials/google_credentials.json",
           spreadsheet_id="YOUR_GOOGLE_SHEET_ID",
           worksheet_name="Лист4",
       )
       
       await client.append_row({
           "client_name": "Тест",
           "client_phone": "+79991234567",
           "status": "тест",
       })
       
       print("✅ Success!")
   
   asyncio.run(test())
   ```

3. **Проверка в Google Sheets**:
   Открой таблицу, проверь что строка добавилась в Лист4

## Частые ошибки

### `FileNotFoundError: credentials/google_credentials.json`

**Решение**: Убедись что credentials.json в нужной папке:
```bash
ls -la credentials/google_credentials.json
```

### `gspread.exceptions.APIError: Insufficient Permission`

**Решение**: 
1. Открой Google Sheets
2. Нажми "Share"
3. Добавь email из credentials.json (`client_email`)
4. Дай права "Editor"

### `WorksheetNotFound: Лист4`

**Решение**: Лист создастся автоматически при первой записи

## Интеграция с Policy Engine

Policy Engine автоматически вызывает notifier когда диалог завершён:

```python
# В policy_engine/main.py
if is_complete and config.notifier.enabled:
    from src.notifier.sheets_client import GoogleSheetsClient
    from src.notifier.formatter import format_slots_for_sheets
    
    # Async запись (не блокирует диалог)
    asyncio.create_task(
        sheets_client.append_row(format_slots_for_sheets(updated_slots))
    )
```

## Безопасность

- ❌ **НЕ коммитить** `credentials.json` в Git
- ✅ `credentials/` в `.gitignore`
- ✅ Использовать Service Account (не personal account)
- ✅ Минимальные права доступа (только к нужной таблице)

## Ссылка на таблицу

https://docs.google.com/spreadsheets/d/YOUR_GOOGLE_SHEET_ID/edit

