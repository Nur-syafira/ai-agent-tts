"""
Tests for dialog slots.
"""

import pytest
from src.policy_engine.slots import DialogSlots
from pydantic import ValidationError


class TestDialogSlots:
    """Tests for DialogSlots model."""

    def test_phone_validation_success(self):
        """Test phone number validation with valid input."""
        slots = DialogSlots(client_phone="+79991234567")
        assert slots.client_phone == "+79991234567"
        
        slots = DialogSlots(client_phone="89991234567")
        assert slots.client_phone == "+79991234567"
        
        slots = DialogSlots(client_phone="9991234567")
        assert slots.client_phone == "+79991234567"

    def test_phone_validation_failure(self):
        """Test phone number validation with invalid input."""
        with pytest.raises(ValidationError):
            DialogSlots(client_phone="123")  # Слишком короткий

    def test_is_complete(self):
        """Test is_complete method."""
        # Incomplete slots
        slots = DialogSlots(client_name="Иван")
        assert not slots.is_complete()
        
        # Complete slots
        slots = DialogSlots(
            client_name="Иван Иванов",
            client_phone="+79991234567",
            study_type="МРТ головного мозга",
            appointment_date="2025-10-25",
            appointment_time="15:00",
            client_age=35,
            client_weight=75.0,
        )
        assert slots.is_complete()

    def test_to_sheets_row(self, sample_dialog_slots):
        """Test conversion to sheets row."""
        row = sample_dialog_slots.to_sheets_row()
        
        assert row["client_name"] == "Иван Иванов"
        assert row["client_phone"] == "+79991234567"
        assert row["status"] == "записан"
        assert "timestamp" in row

