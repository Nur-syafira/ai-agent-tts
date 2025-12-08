# GitHub Repository Setup Guide

This file helps you set up the repository properly on GitHub.

## Repository Settings

### Basic Information
- **Name**: `ai-agent-tts` or `low-latency-voice-ai`
- **Description**: `Low-latency voice AI agent platform with streaming ASR/TTS, FSM-based dialog management, and microservices architecture. Built with FastAPI, LangGraph, vLLM, and F5-TTS.`
- **Visibility**: Public
- **License**: Apache-2.0

### Topics (Tags)
Add these topics to make your repository discoverable:

```
voice-ai
speech-recognition
text-to-speech
conversational-ai
low-latency
streaming-asr
fsm
langgraph
vllm
fastapi
microservices
python
ai
nlp
asr
tts
real-time
production-ready
```

### Website (optional)
If you create a website later, add it here.

### Social Preview
Upload a nice image (1200x630px) showing your project architecture or logo.

## README Best Practices

Your README should:
1. ✅ Start with a clear one-liner
2. ✅ Explain the problem and solution
3. ✅ Show architecture diagram
4. ✅ Include quick start (5 lines)
5. ✅ Link to examples
6. ✅ Show badges (build status, license, etc.)

## Badges to Add

Add these badges to your README.md:

```markdown
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Code style](https://img.shields.io/badge/code%20style-ruff-black.svg)
```

## Repository Features to Enable

1. ✅ **Issues** - Enable for bug reports and feature requests
2. ✅ **Discussions** - Enable for community questions
3. ✅ **Wiki** - Optional, can be useful for detailed docs
4. ✅ **Projects** - Optional, for project management

## Branch Protection

Set up branch protection for `main`:
- Require pull request reviews
- Require status checks to pass
- Require branches to be up to date

## GitHub Actions

Your CI/CD workflow should:
- Run on push to main
- Run on pull requests
- Check code quality (ruff, pyright)
- Run tests (pytest)
- Build Docker images (optional)

