"""
Tests for FSM (Finite State Machine).
"""

import pytest
from src.policy_engine.fsm import DialogFSM, DialogState
from src.policy_engine.slots import DialogSlots


class TestDialogFSM:
    """Tests for DialogFSM."""

    def test_initial_state(self):
        """Test FSM starts in START state."""
        fsm = DialogFSM()
        assert fsm.current_state == DialogState.START

    def test_transition(self):
        """Test state transition."""
        fsm = DialogFSM()
        fsm.transition(DialogState.GREETING)
        assert fsm.current_state == DialogState.GREETING

    def test_get_next_state_simple(self):
        """Test getting next state with simple transition."""
        fsm = DialogFSM()
        
        # START -> GREETING (no condition)
        next_state = fsm.get_next_state(DialogSlots(), "")
        assert next_state == DialogState.GREETING

    def test_get_next_state_with_condition(self):
        """Test getting next state with condition."""
        fsm = DialogFSM()
        fsm.current_state = DialogState.ASK_CLIENT_NAME
        
        # Without name - stay in ASK_CLIENT_NAME
        slots = DialogSlots()
        next_state = fsm.get_next_state(slots, "")
        assert next_state == DialogState.ASK_CLIENT_NAME
        
        # With name - move to ASK_SYMPTOMS
        slots = DialogSlots(client_name="Иван")
        next_state = fsm.get_next_state(slots, "")
        assert next_state == DialogState.ASK_SYMPTOMS

    def test_is_terminal_state(self):
        """Test terminal state detection."""
        fsm = DialogFSM()
        
        assert not fsm.is_terminal_state()
        
        fsm.transition(DialogState.END)
        assert fsm.is_terminal_state()
        
        fsm.transition(DialogState.ERROR)
        assert fsm.is_terminal_state()

    def test_reset(self):
        """Test FSM reset."""
        fsm = DialogFSM()
        fsm.transition(DialogState.GREETING)
        fsm.transition(DialogState.ASK_CLIENT_NAME)
        
        fsm.reset()
        assert fsm.current_state == DialogState.START

    def test_get_state_context(self):
        """Test getting state context."""
        fsm = DialogFSM()
        fsm.current_state = DialogState.GREETING
        
        context = fsm.get_state_context()
        assert "center_name" in context
        assert "admin_name" in context

