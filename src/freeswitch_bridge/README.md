# FreeSWITCH Bridge

Интеграция с **FreeSWITCH** через `mod_audio_fork` для real-time аудиостриминга.

## Статус

⚠️ **В разработке**. Требует установки и настройки FreeSWITCH.

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

## TODO

- [ ] Реализовать WebSocket сервер для mod_audio_fork
- [ ] Интегрировать с ASR Gateway
- [ ] Интегрировать с TTS Gateway
- [ ] Реализовать barge-in через двухканальный VAD
- [ ] Acoustic Echo Cancellation (AEC)
- [ ] Тестирование с реальным FreeSWITCH

## Альтернативы

Если FreeSWITCH сложен для MVP, можно использовать:

1. **Twilio Voice API** - проще, но платно
2. **Asterisk + chan_sip** - open-source, похож на FreeSWITCH
3. **WebRTC** - прямое соединение браузер-сервер (для web-демо)

## Ссылки

- [FreeSWITCH Documentation](https://freeswitch.org/confluence/)
- [mod_audio_fork](https://freeswitch.org/confluence/display/FREESWITCH/mod_audio_fork)

