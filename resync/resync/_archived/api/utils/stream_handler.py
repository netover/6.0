"""
Streaming utilities for WebSocket communication.

This module provides classes for handling real-time streaming of agent responses
over WebSocket connections with proper error handling and message formatting.
"""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

# Import a structured logger from the core logging module. The original code
# referenced `resync_new.utils.simple_logger.get_logger`, which is not present
# in this repository. Use the structured logger implementation instead.
from resync.core.structured_logger import get_logger  # type: ignore[attr-defined]

logger = get_logger(__name__)


class AgentResponseStreamer:
    """Handles streaming of agent responses over WebSocket with robust error handling."""

    def __init__(self, websocket: WebSocket):
        """
        Initialize the streamer with a WebSocket connection.

        Args:
            websocket: The WebSocket connection to stream messages to
        """
        self.websocket = websocket
        self.full_response: str = ""

    async def stream_response(self, agent: Any, query: str) -> str:
        """
        Streams agent response to websocket, accumulating full response.

        Attempts streaming first, falls back to non-streaming if not supported.

        Args:
            agent: Agent instance to query
            query: User query string

        Returns:
            Complete accumulated response

        Raises:
            WebSocketDisconnect: If client disconnects during streaming
        """
        try:
            # Try streaming first
            if callable(getattr(agent, "stream", None)):
                await self._handle_streaming_response(agent, query)
            else:
                await self._handle_non_streaming_response(agent, query)

            return self.full_response

        except WebSocketDisconnect:
            logger.info("websocket_disconnected_during_streaming")
            raise
        except Exception as e:
            logger.error("streaming_error", error=str(e), exc_info=True)
            await self._send_error("Ocorreu um erro ao processar sua solicitação.")
            return self.full_response or "Erro no processamento"

    async def _handle_streaming_response(self, agent: Any, query: str) -> None:
        """Handle streaming response from agent."""
        stream_result = agent.stream(query)

        # Check if result is an async iterator
        if self._is_async_iterator(stream_result):
            await self._stream_chunks(stream_result)
            await self._send_stream_end()
        else:
            # Handle non-async-iterator stream results
            response = str(stream_result)
            self.full_response = response
            await self._send_stream_message(response)

    async def _handle_non_streaming_response(self, agent: Any, query: str) -> None:
        """Handle non-streaming response from agent."""
        response = await agent.arun(query)
        self.full_response = str(response)
        await self._send_stream_message(self.full_response)

    async def _stream_chunks(self, stream: AsyncIterator[str]) -> None:
        """Stream response chunks to client."""
        async for chunk in stream:
            chunk_str = str(chunk)
            self.full_response += chunk_str
            await self._send_stream_message(chunk_str, is_chunk=True)

    async def _send_stream_message(self, message: str, is_chunk: bool = False) -> None:
        """Send a stream message to the client."""
        message_type = "stream"
        await self.websocket.send_json(
            {
                "type": message_type,
                "sender": "agent",
                "message": message,
                "is_chunk": is_chunk,
            }
        )

    async def _send_stream_end(self) -> None:
        """Send stream end marker to client."""
        await self.websocket.send_json({"type": "stream_end"})

    async def _send_error(self, message: str) -> None:
        """Send error message to client if possible."""
        try:
            await self.websocket.send_json(
                {
                    "type": "error",
                    "sender": "system",
                    "message": message,
                }
            )
        except Exception as _e:
            logger.debug("failed_to_send_error_message")

    @staticmethod
    def _is_async_iterator(obj) -> bool:
        """Check if object is an async iterator."""
        return hasattr(obj, "__aiter__")


# Backward compatibility wrapper removed - use AgentResponseStreamer directly
