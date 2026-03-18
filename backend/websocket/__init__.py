"""
ChisCode — WebSocket Package
Real-time communication for project generation progress.
"""
from app.websocket.manager import ws_manager, init_websocket_manager, shutdown_websocket_manager

__all__ = [
    "ws_manager",
    "init_websocket_manager",
    "shutdown_websocket_manager"
]
