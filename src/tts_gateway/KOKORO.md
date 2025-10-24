# Kokoro-82M Integration

## Установка

```bash
# Установить kokoro
./venv/bin/pip install kokoro>=0.9.2 misaki[en]

# Установить espeak-ng (требуется для G2P)
sudo apt-get install espeak-ng

# На macOS
brew install espeak-ng
```

## Доступные голоса

Kokoro-82M поддерживает **9 голосов**:

### Женские (American):
- `af_heart` - Heart (рекомендуется) ⭐
- `af_bella` - Bella
- `af_sarah` - Sarah

### Мужские (American):
- `am_adam` - Adam
- `am_michael` - Michael

### Женские (British):
- `bf_emma` - Emma
- `bf_isabella` - Isabella

### Мужские (British):
- `bm_george` - George
- `bm_lewis` - Lewis

## Конфигурация

Редактируй `src/tts_gateway/config.yaml`:

```yaml
primary_tts:
  enabled: true
  device: "cuda"
  sample_rate: 24000
  speed: 1.0
  voice: "af_heart"  # Смени голос здесь
  lang_code: "a"
```

## Русский язык

⚠️ **Kokoro обучена на английском**, но может попытаться произнести русский текст.

Для русского языка:
- Возможны ошибки произношения
- Акцент будет английский
- **Рекомендация**: используй Piper для русского

Можешь экспериментировать с `lang_code`:
- `a` - английский (по умолчанию)
- Другие коды могут работать, но не гарантированно

## Латентность

На RTX 5090:
- **First-audio**: 50-100 мс
- **Streaming**: чанки по мере генерации
- **Faster than Piper**: на 20-30%

## Качество

- 82M параметров
- StyleTTS 2 архитектура
- Качество близко к XTTS, но в 10x быстрее
- Apache 2.0 лицензия

## Пример использования

```python
from src.tts_gateway.streaming import KokoroTTS

# Инициализация
tts = KokoroTTS(
    device="cuda",
    sample_rate=24000,
    voice="af_heart",
    speed=1.0,
)

# Синтез
text = "Hello! This is a test of Kokoro TTS."
audio = tts.synthesize(text)

# Streaming синтез
for chunk in tts.synthesize_streaming(text):
    # Обработка каждого чанка
    pass
```

## Сравнение с Piper

| | Piper | Kokoro-82M |
|---|---|---|
| Латентность | 80-150 мс | 50-100 мс ⚡ |
| Качество | Хорошее | Отличное ⭐ |
| Русский | ✅ Отлично | ⚠️ Ограниченно |
| Лицензия | MIT | Apache 2.0 |
| Голоса | 1-2 | 9 |

## Troubleshooting

### `ModuleNotFoundError: No module named 'kokoro'`

```bash
./venv/bin/pip install kokoro>=0.9.2
```

### `espeak-ng not found`

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# macOS
brew install espeak-ng
```

### Медленная генерация

Убедись что используется GPU:

```python
import torch
print(torch.cuda.is_available())  # Должно быть True
```

## Ссылки

- [Kokoro на HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
- [GitHub](https://github.com/hexgrad/kokoro)
- [Demo](https://hf.co/spaces/hexgrad/Kokoro-TTS)

