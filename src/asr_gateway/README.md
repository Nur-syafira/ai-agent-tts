# ASR Gateway

Сервис потокового распознавания речи с использованием **RealtimeSTT** (faster-whisper large-v3-turbo) и **Silero VAD**.

## Возможности

- ✅ Потоковое распознавание с low-latency (80-150 мс partial transcripts)
- ✅ Silero VAD для endpointing и barge-in detection
- ✅ WebSocket API для real-time стриминга
- ✅ CUDA-only с guard-проверкой GPU при старте
- ✅ OpenTelemetry трейсинг
- ✅ Health/ready endpoints

## Требования

- Python 3.12
- CUDA-enabled GPU (минимум 2 GB VRAM)
- RTX 5090 (рекомендуется)

## Установка

```bash
# Установить зависимости через uv (из корня проекта)
uv sync

# Или с dev-зависимостями
uv sync --group dev
```

## Конфигурация

Редактируйте `config.yaml`:

```yaml
model:
  name: "large-v3-turbo"  # faster-whisper model
  device: "cuda"
  compute_type: "int8_float16"
  language: "ru"

vad:
  enabled: true
  threshold: 0.5
  min_silence_duration_ms: 500

streaming:
  partial_transcript_interval_ms: 120
  audio_chunk_ms: 160
  sample_rate: 16000
```

Переопределение через ENV:

```bash
export ASR_GATEWAY_MODEL_NAME="large-v3"
export ASR_GATEWAY_VAD_THRESHOLD="0.6"
```

## Запуск

```bash
# Из корня проекта через uv
uv run python src/asr_gateway/main.py

# Или через uvicorn
uv run uvicorn src.asr_gateway.main:app --host 0.0.0.0 --port 8001
```

## API

### WebSocket: `/ws/transcribe`

Отправляйте PCM audio chunks (16 kHz, mono, float32):

```python
import websockets
import numpy as np

async with websockets.connect("ws://localhost:8001/ws/transcribe") as ws:
    audio_chunk = np.random.rand(2560).astype(np.float32)  # 160 ms @ 16 kHz
    await ws.send(audio_chunk.tobytes())
    
    response = await ws.recv()
    print(response)  # {"type": "partial", "text": "...", "timestamp": ...}
```

### GET: `/health`

Проверка здоровья сервиса:

```bash
curl http://localhost:8001/health
```

### GET: `/ready`

Проверка готовности:

```bash
curl http://localhost:8001/ready
```

## Приёмка

1. **Запуск без ошибок**:
   ```bash
   uv run python src/asr_gateway/main.py
   ```
   
2. **Health check проходит**:
   ```bash
   curl http://localhost:8001/health | jq
   ```

3. **GPU обнаружен и используется**:
   Проверьте логи: `"device": "cuda"`, `"CUDA check passed"`

4. **Латентность partial transcripts ≤ 150 мс**:
   Проверяется через OpenTelemetry traces (Jaeger UI)

## Частые ошибки

### `RuntimeError: CUDA is required but not available`

**Решение**: 
- Проверьте `nvidia-smi`
- Проверьте PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Переустановите PyTorch с CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### `FileNotFoundError: config.yaml`

**Решение**: Убедитесь, что запускаете из корня проекта или что `config.yaml` находится в `src/asr_gateway/`

### Медленные partial transcripts (> 200 мс)

**Решение**:
- Проверьте `beam_size: 1` в конфиге
- Проверьте `compute_type: "int8_float16"`
- Убедитесь, что GPU не загружен другими процессами: `nvidia-smi`

## Метрики

- **Partial transcript latency**: 80-150 мс (цель)
- **VAD endpointing**: 150-250 мс
- **VRAM usage**: ~3 GB (large-v3-turbo INT8)

## Логи

Структурированные JSON-логи в stdout:

```json
{
  "timestamp": "2025-10-24T19:30:00.123Z",
  "level": "INFO",
  "service": "asr_gateway",
  "message": "StreamingASR initialized successfully",
  "context": {"model": "large-v3-turbo", "device": "cuda"}
}
```

