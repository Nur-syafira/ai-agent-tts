# TTS Gateway

Сервис синтеза речи с **F5-TTS** для русского языка.

## Возможности

- ✅ F5-TTS для русского языка (автоматическая расстановка ударений через ruaccent) ⚡
- ✅ Пререндер частых фраз (100-150 мс экономии)
- ✅ Redis + файловый кэш
- ✅ Streaming output
- ✅ Low-latency (50-120 мс first-audio)

## Установка

### F5-TTS для русского языка

Модель F5-TTS должна быть размещена в директории `models/F5-tts/` в корне проекта.

Ожидаемые файлы модели:
- `model_last.pt` или `model_last_inference.safetensors`

```bash
# Зависимости уже добавлены в pyproject.toml
# Установка через uv: uv sync
```

## Конфигурация

Редактируйте `config.yaml`:

```yaml
# F5-TTS для русского языка
f5_tts:
  enabled: true
  model_path: "models/F5-tts"  # Локальный путь к модели
  device: "cuda"  # Используем GPU на RTX 5090
  sample_rate: 24000
  use_stress_marks: true  # Автоматическая расстановка ударений

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
# Из корня проекта через uv
uv run python src/tts_gateway/main.py
```

Сервис будет доступен на `http://localhost:8002`.

## API

### `POST /synthesize`

Синтезирует речь из текста.

**Request:**
```json
{
  "text": "Добрый день! Как я могу вам помочь?",
  "use_fallback": false
}
```

**Response:**
- Content-Type: `application/octet-stream`
- Body: PCM аудио (float32, mono, 24 kHz)
- Headers:
  - `X-Sample-Rate`: частота дискретизации
  - `X-Channels`: количество каналов (1)
  - `X-Format`: формат аудио (float32)

### `GET /health`

Проверка здоровья сервиса.

### `GET /metrics`

Prometheus метрики.

## Логика работы

- **Русский текст** → F5-TTS
- **Пререндер** → кэш частых фраз для мгновенного ответа
- **Streaming** → оптимизированные чанки для меньшей задержки первого аудио

## Производительность

- **First-audio latency** (F5-TTS): 50-150 мс
- **Sample rate**: 24000 Hz
- **VRAM**: ~1 GB для F5-TTS модели

## Troubleshooting

### `RuntimeError: F5-TTS initialization failed`

Проверьте:
- CUDA доступен: `python -c "import torch; print(torch.cuda.is_available())"`
- Модель загружается: проверьте интернет-соединение для HuggingFace
- VRAM достаточно: минимум 1 GB свободной VRAM

### `FileNotFoundError: F5-TTS model not found`

Проверьте:
- Модель находится в `models/F5-tts/` относительно корня проекта
- В директории есть файлы `model_last.pt` или `model_last_inference.safetensors`
- Путь в конфигурации указан правильно: `model_path: "models/F5-tts"`

## Ссылки

- [F5-TTS](https://github.com/SWivid/F5-TTS)
