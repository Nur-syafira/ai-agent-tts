"""
Pytest configuration and fixtures.
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_audio_chunk():
    """Sample audio chunk for testing (160 ms @ 16 kHz)."""
    import numpy as np
    return np.random.rand(2560).astype(np.float32)


@pytest.fixture
def sample_dialog_slots():
    """Sample dialog slots for testing."""
    from src.policy_engine.slots import DialogSlots
    
    return DialogSlots(
        client_name="Иван Иванов",
        client_phone="+79991234567",
        symptoms="Боль в голове",
        study_type="МРТ головного мозга",
        appointment_date="2025-10-25",
        appointment_time="15:00",
        client_age=35,
        client_weight=75.0,
    )

