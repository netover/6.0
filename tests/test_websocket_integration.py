import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from resync.app_factory import ApplicationFactory
from resync.core.types.app_state import EnterpriseState


@pytest.fixture
def app():
    factory = ApplicationFactory()
    app = factory.create_application()

    # Mock Enterprise State dependencies
    mock_agent = AsyncMock()
    mock_agent.name = "Test Agent"
    mock_agent.arun.return_value = "Mocked Response"

    mock_manager = AsyncMock()
    mock_manager.get_agent.return_value = mock_agent
    mock_manager.get_all_agents.return_value = []
    mock_manager.get_agent_config.return_value = None

    # Mock HybridRouter response
    mock_router = AsyncMock()
    mock_router_result = MagicMock()
    mock_router_result.response = "Routed Response"
    mock_router_result.routing_mode = MagicMock()
    mock_router_result.routing_mode.value = "agentic"
    mock_router_result.intent = "status"
    mock_router_result.confidence = 0.9
    mock_router_result.handler = "AgenticHandler"
    mock_router_result.tools_used = ["get_tws_status"]
    mock_router_result.entities = {"workstation": ["TWS_PROD"]}
    mock_router_result.trace_id = "test-trace-id"
    mock_router_result.requires_approval = False
    mock_router_result.approval_id = None

    mock_router.route.return_value = mock_router_result

    # Mock ContextStore
    mock_kg = AsyncMock()

    # Create enterprise state
    # We must explicitly set all fields because of slots=True
    state = EnterpriseState(
        connection_manager=MagicMock(),
        knowledge_graph=mock_kg,
        tws_client=MagicMock(),
        agent_manager=mock_manager,
        hybrid_router=mock_router,
        idempotency_manager=MagicMock(),
        llm_service=MagicMock(),
        file_ingestor=MagicMock(),
        a2a_handler=MagicMock(),
        skill_manager=MagicMock(),
        startup_complete=True,
        redis_available=False,
        domain_shutdown_complete=False,
    )

    app.state.enterprise_state = state

    return app


def test_websocket_chat_flow(app):
    client = TestClient(app)

    # Mock authentication service to always verify
    with patch("resync.api.auth.service.AuthService.verify_token", return_value=True):
        # Using a dummy token and agent_id
        with client.websocket_connect("/ws/tws-general?token=valid-token") as websocket:
            # 1. Receive Welcome Message (System)
            welcome = websocket.receive_json()
            assert welcome["type"] == "system"
            assert "Conectado ao agente" in welcome["message"]

            # 2. Send Message
            websocket.send_text("Olá, como está o TWS?")

            # 3. Receive Echo/User Message (per _handle_agent_interaction)
            user_msg = websocket.receive_json()
            assert user_msg["type"] == "message"
            assert user_msg["sender"] == "user"
            assert user_msg["message"] == "Olá, como está o TWS?"

            # 4. Receive Agent Response
            agent_msg = websocket.receive_json()
            assert agent_msg["type"] == "message"
            assert agent_msg["sender"] == "agent"
            assert agent_msg["message"] == "Routed Response"
            assert agent_msg["metadata"]["intent"] == "status"
            assert agent_msg["metadata"]["confidence"] == 0.9
            assert "get_tws_status" in agent_msg["metadata"]["tools_used"]

            # Verify trace ID propagation (correlation_id in payload)
            assert "correlation_id" in agent_msg


def test_websocket_auth_failure(app):
    client = TestClient(app)

    # Mock authentication service to fail
    with patch("resync.api.auth.service.AuthService.verify_token", return_value=False):
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/ws/tws-general?token=invalid"):
                pass
        assert excinfo.value.code == 1008


def test_websocket_invalid_agent(app):
    client = TestClient(app)

    # Mock agent manager to return None (agent not found)
    app.state.enterprise_state.agent_manager.get_agent.return_value = None

    with patch("resync.api.auth.service.AuthService.verify_token", return_value=True):
        with client.websocket_connect(
            "/ws/non-existent-agent?token=valid"
        ) as websocket:
            # 1. First receive the error message sent by the server
            err_msg = websocket.receive_json()
            assert err_msg["type"] == "error"
            assert "não encontrado" in err_msg["message"]

            # 2. Then expect the disconnect
            with pytest.raises(WebSocketDisconnect) as excinfo:
                websocket.receive_json()
            assert excinfo.value.code == 1008
