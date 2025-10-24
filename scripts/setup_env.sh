#!/bin/bash

# ======================================
# System Environment Setup Script
# ======================================
# Настраивает GPU persistence mode и CPU governor для максимальной производительности.

set -e

echo "========================================="
echo "Sales Agent - System Setup"
echo "========================================="

# Проверка root прав
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  This script requires root privileges"
    echo "   Please run with sudo:"
    echo "   sudo ./scripts/setup_env.sh"
    exit 1
fi

echo ""
echo "🔧 Setting up GPU..."

# GPU Persistence Mode
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -pm 1
    echo "✅ GPU persistence mode enabled"
    
    # Показываем информацию о GPU
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found. Skipping GPU setup."
fi

echo ""
echo "🔧 Setting up CPU..."

# CPU Governor = performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -f "$cpu" ]; then
        echo "performance" > "$cpu"
    fi
done

echo "✅ CPU governor set to 'performance'"

# Показываем текущее состояние
echo ""
echo "Current CPU governor:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

echo ""
echo "========================================="
echo "✅ System setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Create venv: python3.12 -m venv venv"
echo "2. Install deps: ./venv/bin/pip install -r requirements.txt"
echo "3. Setup .env: cp .env.example .env"
echo "4. Run services!"

