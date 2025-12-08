# FreeSWITCH Bridge

Интеграция с **FreeSWITCH** через `mod_audio_fork` для real-time аудиостриминга.

## Статус

✅ **Реализовано**. Требует установки и настройки FreeSWITCH для тестирования.

## Архитектура

```
SIP вызов → FreeSWITCH → mod_audio_fork → WebSocket
                                              ↓
                                         ASR Gateway
                                              ↓
                                        Policy Engine
                                              ↓
                                         TTS Gateway
                                              ↓
                                         FreeSWITCH → Клиент
```

## Требования

- FreeSWITCH 1.10.x
- mod_audio_fork (или mod_audio_stream)
- Двухканальное аудио (входящее + исходящее)

## Конфигурация FreeSWITCH

### 1. Установка mod_audio_fork

```bash
# В FreeSWITCH source
cd /usr/src/freeswitch
./configure --enable-core-pgsql-support
make mod_audio_fork
make mod_audio_fork-install
```

### 2. Включение модуля

Добавить в `modules.conf.xml`:

```xml
<load module="mod_audio_fork"/>
```

### 3. Конфигурация dialplan

Добавить в `dialplan/default.xml`:

```xml
<extension name="mri_bot">
  <condition field="destination_number" expression="^(MRI_BOT)$">
    <action application="answer"/>
    <action application="audio_fork" data="ws://localhost:8004/ws/audio start"/>
    <action application="park"/>
  </condition>
</extension>
```

## WebSocket Protocol

### Входящий аудио (от FreeSWITCH)

```json
{
  "type": "audio",
  "direction": "input",
  "format": "L16",
  "sample_rate": 16000,
  "channels": 1,
  "data": "<base64 encoded PCM>"
}
```

### Исходящий аудио (в FreeSWITCH)

```json
{
  "type": "audio",
  "direction": "output",
  "format": "L16",
  "sample_rate": 16000,
  "channels": 1,
  "data": "<base64 encoded PCM>"
}
```

## Barge-In Detection

**Двухканальный VAD**:
- VAD на входящем канале → клиент заговорил
- VAD на исходящем канале → агент говорит
- **Barge-in**: если клиент заговорил ПОКА агент говорит → останавливаем TTS

## Реализовано

- [x] WebSocket сервер для mod_audio_fork (`/ws/audio`)
- [x] Интеграция с ASR Gateway (WebSocket клиент)
- [x] Интеграция с TTS Gateway (HTTP POST)
- [x] Интеграция с Policy Engine (HTTP POST)
- [x] Обработка двунаправленного аудио потока
- [x] Управление сессиями звонков
- [ ] Barge-in detection (двухканальный VAD) - базовая структура готова
- [ ] Acoustic Echo Cancellation (AEC) - будущая доработка
- [ ] Тестирование с реальным FreeSWITCH

## Запуск

```bash
# Из корня проекта через uv
uv run python src/freeswitch_bridge/main.py

# Или через uvicorn
uv run uvicorn src.freeswitch_bridge.main:app --host 0.0.0.0 --port 8004
```

## API

### WebSocket: `/ws/audio`

Endpoint для FreeSWITCH mod_audio_fork.

**Входящие сообщения (от FreeSWITCH):**
```json
{
  "type": "audio",
  "direction": "input",
  "format": "L16",
  "sample_rate": 16000,
  "channels": 1,
  "data": "<base64 encoded PCM>"
}
```

**Исходящие сообщения (в FreeSWITCH):**
```json
{
  "type": "audio",
  "direction": "output",
  "format": "L16",
  "sample_rate": 16000,
  "channels": 1,
  "data": "<base64 encoded PCM>"
}
```

### GET `/calls`

Список активных звонков:
```json
{
  "active_calls": [
    {
      "call_id": "uuid-123",
      "session_id": "call-uuid-123",
      "is_speaking": false
    }
  ]
}
```

## Поток данных

1. **Входящий звонок** → FreeSWITCH подключается к `/ws/audio`
2. **Входящий аудио** → ASR Gateway (WebSocket) → транскрипт
3. **Транскрипт** → Policy Engine (HTTP POST `/dialog`) → ответ агента
4. **Ответ агента** → TTS Gateway (HTTP POST `/synthesize`) → аудио
5. **Исходящий аудио** → FreeSWITCH (WebSocket) → клиент слышит

## Конфигурация

Редактируйте `config.yaml`:

```yaml
services:
  asr_ws_url: "http://localhost:8001"
  tts_http_url: "http://localhost:8002"
  policy_http_url: "http://localhost:8003"

audio:
  sample_rate: 16000
  channels: 1
  format: "L16"
  chunk_size_ms: 200

barge_in:
  enabled: true
  vad_threshold: 0.5
  min_speech_duration_ms: 100

server:
  host: "0.0.0.0"
  port: 8004
  max_connections: 100
```

## Приёмка

1. **Запуск без ошибок**:
   ```bash
   uv run python src/freeswitch_bridge/main.py
   ```

2. **Health check**:
   ```bash
   curl http://localhost:8004/health
   ```

3. **Список звонков**:
   ```bash
   curl http://localhost:8004/calls
   ```

4. **Тестирование с FreeSWITCH**:
   - Настроить FreeSWITCH dialplan (см. выше)
   - Совершить SIP звонок на номер `MRI_BOT`
   - Проверить логи FreeSWITCH Bridge

## Альтернативы

Если FreeSWITCH сложен для MVP, можно использовать:

1. **Twilio Voice API** - проще, но платно
2. **Asterisk + chan_sip** - open-source, похож на FreeSWITCH
3. **WebRTC** - прямое соединение браузер-сервер (для web-демо)

## Ссылки

- [FreeSWITCH Documentation](https://freeswitch.org/confluence/)
- [mod_audio_fork](https://freeswitch.org/confluence/display/FREESWITCH/mod_audio_fork)

