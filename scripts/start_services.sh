#!/usr/bin/env bash
# Скрипт для запуска всех сервисов и тестирования диалога

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "🚀 Sales Agent - Запуск сервисов"
echo "=========================================="
echo ""

# Проверка Redis
echo "1️⃣  Проверка Redis..."
if docker ps | grep -q redis; then
    echo "   ✅ Redis запущен"
else
    echo "   ⚠️  Redis не запущен, запускаю..."
    docker-compose up -d redis
    sleep 2
fi

# Проверка vLLM
echo ""
echo "2️⃣  Проверка vLLM сервера..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "   ✅ vLLM сервер запущен"
else
    echo "   ⚠️  vLLM сервер не запущен"
    echo "   Запустите в отдельном терминале:"
    echo "   vllm serve models/Qwen3-16B-A3B-abliterated-AWQ \\"
    echo "     --host 0.0.0.0 --port 8000 \\"
    echo "     --quantization awq \\"
    echo "     --enable-chunked-prefill --enable-prefix-caching"
    echo ""
    read -p "Нажмите Enter когда vLLM будет запущен..."
fi

# Проверка Policy Engine
echo ""
echo "3️⃣  Проверка Policy Engine..."
if curl -s http://localhost:8003/health > /dev/null 2>&1; then
    echo "   ✅ Policy Engine запущен"
else
    echo "   ⚠️  Policy Engine не запущен"
    echo "   Запускаю Policy Engine в фоне..."
    uv run python src/policy_engine/main.py > /tmp/policy_engine.log 2>&1 &
    POLICY_PID=$!
    echo "   Policy Engine запущен (PID: $POLICY_PID)"
    echo "   Логи: /tmp/policy_engine.log"
    sleep 3
    
    # Проверяем что запустился
    if curl -s http://localhost:8003/health > /dev/null 2>&1; then
        echo "   ✅ Policy Engine готов"
    else
        echo "   ❌ Policy Engine не запустился, проверьте логи"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "✅ Все сервисы готовы!"
echo "=========================================="
echo ""
echo "Теперь можно запустить симуляцию диалога:"
echo ""
echo "  uv run python scripts/test_dialog_performance.py"
echo ""
echo "Или с конкретным сценарием:"
echo ""
echo "  uv run python scripts/test_dialog_performance.py --scenario-name basic_success"
echo ""

