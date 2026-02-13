import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI
from resync.app_factory import ApplicationFactory
from resync.core.exceptions import ConfigurationError

@pytest.mark.asyncio
async def test_startup_timeout_triggered():
    """Verify that a slow startup check triggers the global timeout."""
    factory = ApplicationFactory()
    app = FastAPI()
    
    # Mock settings with a very short timeout
    # We patch settings in app_factory because it's imported at module level there
    with patch("resync.app_factory.settings") as mock_settings:
        mock_settings.startup_timeout = 0.5  # 500ms
        
        # Simula um check que demora 1 segundo
        async def slow_startup_checks(*args, **kwargs):
            await asyncio.sleep(1.0)
            return {"overall_health": True}
            
        # Patch the SOURCE of run_startup_checks because it's imported locally in lifespan
        with patch("resync.core.startup.run_startup_checks", side_effect=slow_startup_checks):
            with pytest.raises(ConfigurationError, match="startup exceeded 0.5s timeout"):
                async with factory.lifespan(app):
                    pass

@pytest.mark.asyncio
async def test_startup_success_within_time():
    """Verify that startup completes if checks are fast."""
    factory = ApplicationFactory()
    app = FastAPI()
    
    with patch("resync.app_factory.settings") as mock_settings:
        mock_settings.startup_timeout = 2.0
        
        mock_result = {"overall_health": True, "critical_services": [], "strict": False}
        
        # Patch the source modules
        with patch("resync.core.startup.run_startup_checks", AsyncMock(return_value=mock_result)):
            with patch("resync.core.startup.enforce_startup_policy"):
                with patch("resync.core.wiring.init_domain_singletons"):
                    with patch("resync.app_factory.enterprise_state_from_app") as mock_st_getter:
                        mock_st = MagicMock()
                        mock_st_getter.return_value = mock_st
                        
                        with patch("resync.core.tws_monitor.get_tws_monitor", AsyncMock()):
                            with patch("resync.api_gateway.container.setup_dependencies"):
                                # For internal methods of factory, we use patch.object
                                with patch.object(factory, "_init_proactive_monitoring", AsyncMock()):
                                    with patch.object(factory, "_init_metrics_collector", AsyncMock()):
                                        with patch.object(factory, "_init_cache_warming", AsyncMock()):
                                            with patch.object(factory, "_init_graphrag", AsyncMock()):
                                                with patch.object(factory, "_init_config_system", AsyncMock()):
                                                    async with factory.lifespan(app):
                                                        assert mock_st.startup_complete is True

@pytest.mark.asyncio
async def test_individual_helper_timeout():
    """Verify that an individual helper's timeout doesn't crash the whole startup."""
    factory = ApplicationFactory()
    app = FastAPI()
    
    async def slow_collector():
        await asyncio.sleep(0.5)
        
    with patch("resync.app_factory.settings") as mock_settings:
        mock_settings.startup_timeout = 5.0
        
        with patch("resync.core.startup.run_startup_checks", AsyncMock(return_value={"overall_health": True})):
            with patch("resync.core.wiring.init_domain_singletons"):
                with patch("resync.app_factory.enterprise_state_from_app", return_value=MagicMock()):
                    with patch("resync.core.tws_monitor.get_tws_monitor", AsyncMock()):
                        with patch("resync.api_gateway.container.setup_dependencies"):
                            # We want to trigger the TimeoutError inside _init_metrics_collector
                            # It has a 10s timeout. We'll wrap the call to make it shorter.
                            
                            # Mock the task creator to be slow
                            async def very_slow_task(*args, **kwargs):
                                await asyncio.sleep(0.5)
                                return MagicMock()
                                
                            with patch("resync.core.task_tracker.create_tracked_task", side_effect=very_slow_task):
                                # We need to mock asyncio.timeout to return a short timeout when it's called with 10
                                original_timeout = asyncio.timeout
                                def mock_timeout(seconds):
                                    if seconds == 10:
                                        return original_timeout(0.1)
                                    return original_timeout(seconds)
                                
                                with patch("asyncio.timeout", side_effect=mock_timeout):
                                    # Should complete without raising TimeoutError to the top
                                    async with factory.lifespan(app):
                                        pass
                                        # If we reach here, the timeout in the helper was caught!
