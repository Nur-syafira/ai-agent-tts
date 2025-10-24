# TTS Gateway

Сервис синтеза речи с **Kokoro-82M** (primary, английский) и **Piper** (fallback, русский).

## Возможности

- ✅ Kokoro-82M TTS (50-100 мс, английский, 9 голосов) ⚡
- ✅ Piper TTS (80-150 мс, русский, fallback)
- ✅ Пререндер частых фраз (100-150 мс экономии)
- ✅ Redis + файловый кэш
- ✅ Streaming output
- ✅ Low-latency (50-120 мс first-audio)

## Установка

### Piper TTS

```bash
# Linux
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
sudo mv piper/piper /usr/local/bin/

# Скачать русскую модель
mkdir -p models
cd models
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json
cd ..
```

### Kokoro-82M

```bash
# Установить kokoro
pip install kokoro>=0.9.2 misaki[en]

# Установить espeak-ng (требуется для G2P)
sudo apt-get install espeak-ng
```

Подробнее: [KOKORO.md](KOKORO.md)

## Конфигурация

Редактируйте `config.yaml`:

```yaml
fallback_tts:
  enabled: true
  model_name: "piper"
  model_path: "models/ru_RU-dmitri-medium.onnx"
  config_path: "models/ru_RU-dmitri-medium.onnx.json"
  sample_rate: 22050
  speed: 1.0

prerender:
  enabled: true
  cache_dir: "cache/tts"
  ttl_seconds: 3600
  common_phrases:
    - "Добрый день!"
    - "Меня зовут администратор медицинского центра."
    # ... другие фразы
```

## Запуск

```bash
./venv/bin/python src/tts_gateway/main.py
```

## API

### POST `/synthesize`

Синтезирует речь:

```bash
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Добрый день!"}' \
  --output audio.raw

# Проиграть аудио
ffplay -f f32le -ar 22050 -ac 1 audio.raw
```

Python:

```python
import httpx
import numpy as np

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8002/synthesize",
        json={"text": "Добрый день!"}
    )
    
    audio = np.frombuffer(response.content, dtype=np.float32)
    # audio shape: (n_samples,), sample_rate=22050
```

## Пререндер

При старте автоматически пререндерятся фразы из `config.yaml`:

```
2025-10-24 19:30:00 - INFO - Prerendering 8 phrases...
2025-10-24 19:30:01 - INFO - Prerendered (1/8): Добрый день!
...
2025-10-24 19:30:05 - INFO - Prerendering completed
```

Пререндеренные фразы отдаются **мгновенно** (< 10 мс) из кэша.

## Кэширование

1. **Redis** (primary cache):
   - TTL: 1 час
   - Автоматическая инвалидация

2. **File cache** (fallback):
   - Папка: `cache/tts/`
   - Формат: `{md5(text)}.pkl`

## Метрики

- **First-audio latency** (Piper): 80-150 мс
- **Cached phrases**: < 10 мс
- **Sample rate**: 22050 Hz (ресемплируется в 16000 для совместимости)

## Приёмка

1. **Запуск без ошибок**:
   ```bash
   ./venv/bin/python src/tts_gateway/main.py
   ```

2. **Синтез работает**:
   ```bash
   curl -X POST http://localhost:8002/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "Тест"}' > test.raw
   
   # Проверить размер файла
   ls -lh test.raw
   ```

3. **Пререндер выполнен**:
   Проверьте логи: `"Prerendering completed"`

4. **Redis кэш работает** (если Redis запущен):
   ```bash
   redis-cli KEYS "tts:*"
   ```

## Частые ошибки

### `FileNotFoundError: Piper model not found`

**Решение**: Скачайте модель по инструкции выше

### `redis.exceptions.ConnectionError`

**Решение**: Redis опционален, сервис будет использовать файловый кэш

### Искажённое аудио

**Решение**: Убедитесь, что используете правильный sample_rate при воспроизведении:
```bash
ffplay -f f32le -ar 22050 -ac 1 audio.raw
```

## Roadmap

- [ ] Интеграция Kokoro-82M через ONNX Runtime
- [ ] True streaming (чанки по мере генерации, не после полного синтеза)
- [ ] Поддержка голосовых профилей
- [ ] Динамическая скорость речи (для более быстрых фраз)

## Ссылки

- [Piper TTS](https://github.com/rhasspy/piper)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

