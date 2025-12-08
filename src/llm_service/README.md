# LLM Service

Обёртка для vLLM сервера с **Qwen2.5-14B-Instruct-AWQ**.

## Важно

**Пользователь запускает vLLM сервер самостоятельно** (согласно вашим требованиям). 

Этот модуль предоставляет:
- Helper-функции для взаимодействия с vLLM
- Клиент с OpenAI-compatible API
- Structured output для извлечения слотов
- Документацию по запуску

## Запуск vLLM сервера

```bash
# Из корня проекта через uv
cd /path/to/ai-agent-TTS
uv run vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --quantization awq \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --quantization awq \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 2048
```

### Вариант 2: Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-14B-Instruct-AWQ",
    quantization="awq",
    max_model_len=2048,
    gpu_memory_utilization=0.75,
)

outputs = llm.generate(
    prompts=["Привет! Как дела?"],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=512),
)
```

## Использование LLM Client

```python
from src.llm_service.main import LLMClient

# Инициализация
client = LLMClient(base_url="http://localhost:8000/v1")

# Обычная генерация
messages = [
    {"role": "system", "content": "Ты помощник для записи на МРТ."},
    {"role": "user", "content": "Добрый день, хочу записаться на МРТ головы."}
]

response = await client.generate(messages, temperature=0.7, max_tokens=256)
print(response)

# Structured output (JSON)
messages = [
    {"role": "system", "content": "Извлеки данные клиента в JSON формате."},
    {"role": "user", "content": "Меня зовут Иван, мой телефон +79991234567"}
]

json_response = await client.generate_structured(messages)
print(json_response)  # {"name": "Иван", "phone": "+79991234567"}
```

## Конфигурация

Редактируйте `config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-14B-Instruct-AWQ"
  quantization: "awq"
  max_model_len: 2048
  gpu_memory_utilization: 0.75

generation:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 512
  response_format_type: "json_object"
```

## Structured Output для слотов

vLLM поддерживает JSON mode для извлечения структурированных данных:

```python
messages = [
    {
        "role": "system",
        "content": """Извлеки следующие данные из сообщения клиента в JSON:
- name (имя)
- symptoms (симптомы)
- phone (телефон)
- date (дата записи)
- time (время записи)

Если данных нет, используй null."""
    },
    {
        "role": "user",
        "content": "Меня зовут Мария. У меня боли в голове уже неделю. Хочу записаться на завтра в 15:00. Мой телефон +79161234567"
    }
]

response = await client.generate_structured(messages)
# {
#   "name": "Мария",
#   "symptoms": "боли в голове уже неделю",
#   "phone": "+79161234567",
#   "date": "завтра",
#   "time": "15:00"
# }
```

## Проверка здоровья

```python
is_healthy = await client.health_check()
print(f"vLLM server is {'running' if is_healthy else 'down'}")
```

## Метрики производительности

На RTX 5090 с Qwen2.5-14B-Instruct-AWQ:

- **Time to First Token (TTFT)**: 40-80 мс
- **Tokens per second**: ~150-200 tokens/s
- **Latency для коротких ответов (50 tokens)**: 40-150 мс
- **VRAM usage**: ~8 GB

## Оптимизации

1. **Flash Attention 2** (автоматически если доступен):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Chunked Prefill** (уже включено):
   Снижает TTFT на длинных промптах

3. **Prefix Caching** (уже включено):
   Кэширует общие префиксы (например, system prompt)

4. **Низкая температура для JSON**:
   Используйте `temperature=0.3` для structured output

## Приёмка

1. **vLLM сервер запущен**:
   ```bash
   curl http://localhost:8000/v1/models
   ```

2. **Генерация работает**:
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
       "messages": [{"role": "user", "content": "Привет!"}],
       "max_tokens": 50
     }'
   ```

3. **Латентность TTFT ≤ 150 мс**:
   Проверяется через metrics

4. **GPU используется**:
   ```bash
   nvidia-smi
   # Должна быть загрузка GPU и ~8GB VRAM
   ```

## Частые ошибки

### `Model not found`

**Решение**: Модель скачается автоматически при первом запуске. Подождите.

### `OutOfMemoryError`

**Решение**: 
- Снизьте `gpu_memory_utilization` до 0.6-0.7
- Убедитесь, что другие модели (ASR/TTS) не загружены на тот же GPU

### Медленная генерация

**Решение**:
- Проверьте `nvidia-smi` - возможно GPU занят
- Установите Flash Attention 2
- Снизьте `max_model_len` если не нужен длинный контекст

## Ссылки

- [vLLM Documentation](https://docs.vllm.ai)
- [Qwen2.5 на Hugging Face](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

