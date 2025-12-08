# Low-Latency Voice AI Agent

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![Type checking](https://img.shields.io/badge/type%20checking-pyright-yellow.svg)](https://github.com/microsoft/pyright)

> Modular platform for building voice AI agents with minimal latency (250-600 ms E2E)

## Problem

Building production-ready voice AI agents requires solving multiple complex problems simultaneously:
- **Low latency** (mouth→ear < 600 ms) for natural conversation
- **Streaming audio processing** without buffering
- **Dialog management** via FSM with structured data extraction
- **Scalability** through microservices architecture
- **Monitoring** E2E metrics for debugging

Most existing solutions are either too slow (>1s latency), monolithic, or don't provide a ready architecture for production.

## Solution

A modular platform based on microservices with optimized components:
- **ASR Gateway**: streaming speech recognition (RealtimeSTT + faster-whisper, 80-150 ms)
- **TTS Gateway**: speech synthesis with caching (F5-TTS, 50-150 ms)
- **LLM Service**: inference via vLLM with AWQ quantization (40-150 ms)
- **Policy Engine**: dialog management via LangGraph FSM + Pydantic slots
- **Monitoring**: OpenTelemetry for E2E tracing

The architecture allows running all components locally on a single GPU (12+ GB VRAM) or scaling horizontally.

## Features

- ✔ Modular architecture
- ✔ Highly extensible
- ✔ Production-ready
- ✔ Low latency (250-600 ms E2E)
- ✔ Streaming ASR/TTS
- ✔ FSM-based dialog management
- ✔ Structured data extraction
- ✔ OpenTelemetry monitoring

## Architecture

```
SIP/PBX → FreeSWITCH (mod_audio_fork)
   ├─(L16 PCM, 16 kHz, 160 ms)→ ASR Gateway
   │                              ↓
   │                         Policy Engine (FSM + Pydantic Slots)
   │                              ↓
   │                         LLM Service (vLLM)
   │                              ↓
   └←(PCM chunks, 200-300 ms)←  TTS Gateway
                                  ↓
                            Google Sheets Notifier
```

See [OVERVIEW.md](OVERVIEW.md) for detailed architecture.

## Target Metrics

| Component | Latency | VRAM |
|-----------|---------|------|
| ASR partial | 80-150 ms | ~3 GB |
| LLM inference | 40-150 ms | ~6 GB (MoE model) |
| TTS first-audio | 50-120 ms | ~1 GB |
| **E2E (mouth→ear)** | **250-600 ms** | **~10 GB** |

## Tech Stack

- **ASR**: RealtimeSTT + faster-whisper large-v3-turbo
- **VAD**: Silero VAD v5
- **LLM**: Qwen3-16B-A3B-abliterated-AWQ (vLLM, MoE architecture)
- **TTS**: F5-TTS (Russian)
- **Policy**: LangGraph FSM + Pydantic slots
- **Storage**: Redis (sessions)
- **Sheets**: gspread-asyncio
- **Monitoring**: OpenTelemetry + Jaeger

## Requirements

- **OS**: Linux (Ubuntu 22.04+)
- **Python**: 3.12
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip
- **GPU**: CUDA-enabled (RTX 5090 recommended, minimum 12 GB VRAM)
- **RAM**: 64 GB (recommended)
- **CPU**: AMD Ryzen 9 9950X3D or equivalent

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/ai-agent-TTS.git
cd ai-agent-TTS
uv sync

# 2. Download models
uv run python scripts/download_models.py

# 3. Configure environment
cp .env.example .env
# Edit .env (set your paths)

# 4. Start services
docker-compose up -d redis jaeger
uv run python scripts/start_services.sh

# 5. Health check
uv run python scripts/health_check.py
```

**Minimal usage example:**

```python
from src.policy_engine.main import PolicyEngine

# Initialize (automatically connects to ASR/TTS/LLM)
engine = PolicyEngine()

# Process user message
response = await engine.process_message(
    session_id="test-123",
    user_message="Hello, I want to book an appointment"
)

print(response.agent_message)  # Agent response
print(response.slots)  # Extracted data
```

See [QUICK_START.md](QUICK_START.md) and [examples/](examples/) for details.

## Examples

See `examples/` folder — working code ready to run:

- `examples/basic_dialog.py` — basic dialog
- `examples/custom_fsm.py` — FSM customization
- `examples/custom_prompts.py` — prompt configuration

## Extensibility

The platform is easily extensible through configuration:

```python
# Custom FSM
from src.policy_engine.fsm import DialogFSM, DialogState

class MyFSM(DialogFSM):
    def _build_transitions(self):
        # Your transition logic
        return [...]
```

```yaml
# config.yaml
llm:
  model_name: "your-model"
  temperature: 0.7
```

## Documentation

- [OVERVIEW.md](OVERVIEW.md) — architecture and data flows
- [QUICK_START.md](QUICK_START.md) — detailed setup guide
- [src/asr_gateway/README.md](src/asr_gateway/README.md) — ASR service
- [src/llm_service/README.md](src/llm_service/README.md) — LLM service
- [src/tts_gateway/README.md](src/tts_gateway/README.md) — TTS service
- [src/policy_engine/README.md](src/policy_engine/README.md) — Policy Engine
- [src/notifier/README.md](src/notifier/README.md) — Google Sheets integration
- [src/freeswitch_bridge/README.md](src/freeswitch_bridge/README.md) — FreeSWITCH integration

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Tests with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Simulate full dialog
uv run python scripts/simulate_dialog.py

# Simulate with specific scenario
uv run python scripts/simulate_dialog.py --scenario scripts/dialog_scenarios.yaml
```

## Monitoring

### Prometheus Metrics

Each service exports Prometheus metrics on `/metrics` endpoint:

- `{service}_requests_total` - total requests
- `{service}_request_latency_seconds` - request latency
- `{service}_active_connections` - active connections
- `{service}_errors_total` - error count

### Jaeger UI

Open http://localhost:16686 to view traces:

- E2E latency (mouth→ear)
- ASR partial latency
- LLM TTFT (Time to First Token)
- TTS TTFA (Time to First Audio)

### Logs

Structured JSON logs to stdout:

```json
{
  "timestamp": "2025-10-24T19:30:00.123Z",
  "level": "INFO",
  "service": "policy_engine",
  "message": "Dialog complete for session abc-123",
  "context": {"session_id": "abc-123", "slots_filled": 12}
}
```

## Docker

The project supports containerization via Docker and docker-compose.

### Build

```bash
# Build Docker image
docker build -t ai-agent-tts:latest .

# Or via docker-compose
docker-compose build
```

### Run

```bash
# Start all services (Redis, Jaeger, ASR Gateway, TTS Gateway, Policy Engine)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Note**: GPU support in Docker requires NVIDIA Container Toolkit setup.

## CI/CD

The project uses GitHub Actions for automated code checks and image builds.

### Workflow includes:

- **Linting**: ruff (formatting and code checks), pyright (type checking)
- **Testing**: pytest with code coverage
- **Security scan**: bandit (code security), pip-audit (dependency vulnerabilities), Trivy (Docker image scanning)
- **Docker build**: automatic image builds on push to main/master

### Local checks

```bash
# Install dev dependencies
uv sync --group dev

# Run linting
uv run ruff check .
uv run ruff format --check .
uv run pyright src/

# Run tests
uv run pytest tests/ -v --cov=src

# Security checks
uv run bandit -r src/
uv run pip-audit
```

### Pre-commit hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run checks manually
uv run pre-commit run --all-files
```

## Security

- ✅ Credentials in `.gitignore` (never committed)
- ✅ Service Account (not personal account) for Google Sheets
- ✅ `.env` not committed (`.env.example` as template)
- ✅ Minimal access rights
- ✅ Security scan in CI/CD (bandit, pip-audit, Trivy)
- ✅ Pre-commit hooks for secret checking (gitleaks)
- ✅ Containers run as non-root user

## 💎 Premium Features

Looking for ready-to-use solutions?

- 🎯 **Domain-Specific Prompts** — Optimized prompts for medical centers, sales, support
- 🚀 **Advanced FSM Templates** — Pre-built dialog scenarios
- ⚡ **Performance Optimization Pack** — Advanced optimization techniques

*Premium features available separately. [Learn more →](mailto:premium@yourdomain.com)*

## 🤝 Consulting & Integration

Need customization or integration?

- Custom FSM development for your domain
- Prompt optimization for your use case
- Production deployment support
- Team training

*[Contact us →](mailto:consulting@yourdomain.com)*

## Roadmap

- [ ] True streaming TTS (not chunks after synthesis)
- [ ] A/B testing prompts
- [ ] Multi-turn context (>10 messages)
- [ ] WebRTC for web demo
- [ ] Horizontal scaling (load balancer)
- [ ] Multi-tenant support
- [ ] Dashboard (Grafana)

## License

Apache 2.0 — see [LICENSE](LICENSE)

## Citation

```bibtex
@software{ai-agent-tts2025,
  title = {Low-Latency Voice AI Agent},
  author = {Mordvinov, Aleksandr},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/ai-agent-TTS}
}
```

## Author

**Aleksandr Mordvinov**

## Acknowledgments

- **Qwen Team** — Qwen2.5 LLM
- **Systran** — faster-whisper
- **Silero Team** — Silero VAD
- **SWivid** — F5-TTS
- **Misha24-10** — F5-TTS Russian model
- **vLLM Team** — vLLM inference engine

---

Made with ❤️ for low-latency voice AI

