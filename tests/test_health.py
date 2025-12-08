"""
Tests for health check.
"""

import pytest
from src.shared.health import HealthChecker


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_initialization(self):
        """Test health checker initialization."""
        checker = HealthChecker("test_service", "0.1.0")
        assert checker.service_name == "test_service"
        assert checker.version == "0.1.0"

    def test_create_router(self):
        """Test creating FastAPI router."""
        checker = HealthChecker("test_service")
        router = checker.create_router()
        
        # Check routes exist
        route_paths = [route.path for route in router.routes]
        assert "/health" in route_paths
        assert "/ready" in route_paths

    @pytest.mark.asyncio
    async def test_check_readiness(self):
        """Test readiness check."""
        checker = HealthChecker("test_service")
        is_ready = await checker._check_readiness()
        assert is_ready is True

    def test_get_system_stats(self):
        """Test getting system stats."""
        stats = HealthChecker.get_system_stats()
        
        assert "cpu_percent" in stats
        assert "memory_percent" in stats
        assert "memory_available_gb" in stats
        assert isinstance(stats["cpu_percent"], float)

