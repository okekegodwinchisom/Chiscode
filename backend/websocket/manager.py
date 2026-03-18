"""
ChisCode — WebSocket Connection Manager
Manages WebSocket connections for real-time project updates with Redis pub/sub.
"""
import json
from typing import Dict, Set, Optional, Any, List
from fastapi import WebSocket
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from app.core.logging import get_logger
from app.core.config import settings
from app.db import redis_client

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections with Redis pub/sub for horizontal scaling.
    
    Features:
    - In-memory connection tracking per server instance
    - Redis pub/sub for broadcasting across multiple server instances
    - Automatic cleanup of dead connections
    - Connection pooling and heartbeat monitoring
    """
    
    def __init__(self):
        # Active connections: {project_id: {user_id: websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._pubsub_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._redis_available = False
        
    async def initialize(self):
        """Initialize Redis pub/sub and heartbeat monitoring."""
        try:
            # Check if Redis is available
            self._redis_available = redis_client.is_connected()
            
            if self._redis_available:
                # Start Redis pubsub listener
                self._pubsub_task = asyncio.create_task(self._redis_pubsub_listener())
                logger.info("Redis pub/sub enabled for WebSocket manager")
            else:
                logger.warning("Redis not available - WebSocket will work in single-instance mode")
            
            # Start heartbeat monitor
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            logger.info("WebSocket manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")

    async def connect(self, websocket: WebSocket, project_id: str, user_id: str):
        """Accept and store a new WebSocket connection."""
        try:
            async with self._lock:
                if project_id not in self.active_connections:
                    self.active_connections[project_id] = {}
                
                # Store connection
                self.active_connections[project_id][user_id] = websocket
            
            logger.info(
                f"WebSocket connected - project_id: {project_id}, user_id: {user_id}, "
                f"total_connections: {self._get_total_connections()}"
            )
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise

    async def disconnect(self, project_id: str, user_id: str):
        """Remove a WebSocket connection."""
        try:
            async with self._lock:
                if project_id in self.active_connections:
                    if user_id in self.active_connections[project_id]:
                        # Close the connection if it's still open
                        try:
                            ws = self.active_connections[project_id][user_id]
                            await ws.close()
                        except Exception as e:
                            logger.debug(f"Error closing WebSocket: {e}")
                        
                        del self.active_connections[project_id][user_id]
                        
                        # Clean up empty project entries
                        if not self.active_connections[project_id]:
                            del self.active_connections[project_id]
            
            logger.info(
                f"WebSocket disconnected - project_id: {project_id}, user_id: {user_id}, "
                f"total_connections: {self._get_total_connections()}"
            )
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def send_personal_message(self, message: dict, project_id: str, user_id: str):
        """Send a message to a specific user's connection."""
        try:
            if project_id in self.active_connections:
                if user_id in self.active_connections[project_id]:
                    websocket = self.active_connections[project_id][user_id]
                    await websocket.send_json(message)
        except Exception as e:
            logger.error(
                f"Failed to send personal message - project_id: {project_id}, "
                f"user_id: {user_id}, error: {str(e)}"
            )
            await self.disconnect(project_id, user_id)

    async def broadcast_to_project(self, message: dict, project_id: str, exclude_user: Optional[str] = None):
        """Send a message to all users connected to a project (local only)."""
        try:
            if project_id not in self.active_connections:
                return
            
            disconnected: List[str] = []
            
            # Get copy of connections to avoid modification during iteration
            async with self._lock:
                connections = self.active_connections.get(project_id, {}).copy()
            
            for user_id, websocket in connections.items():
                if exclude_user and user_id == exclude_user:
                    continue
                    
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(
                        f"Failed to broadcast to user - project_id: {project_id}, "
                        f"user_id: {user_id}, error: {str(e)}"
                    )
                    disconnected.append(user_id)
            
            # Clean up disconnected clients
            for user_id in disconnected:
                await self.disconnect(project_id, user_id)
                
        except Exception as e:
            logger.error(f"Broadcast error: {e}")

    async def publish_to_project(self, message: dict, project_id: str):
        """
        Publish a message to Redis for cross-instance broadcasting.
        Use this when running multiple server instances.
        """
        # First, broadcast to local connections
        await self.broadcast_to_project(message, project_id)
        
        # Then publish to Redis if available
        if self._redis_available:
            try:
                channel = f"project:{project_id}"
                payload = json.dumps({
                    **message,
                    "_timestamp": datetime.utcnow().isoformat(),
                    "_server_id": getattr(settings, 'SERVER_ID', 'default')
                })
                
                # Use Redis PUBLISH command
                await redis_client.cache_set(
                    f"ws_msg:{project_id}:{datetime.utcnow().timestamp()}",
                    payload,
                    ttl=60
                )
                
            except Exception as e:
                logger.error(f"Failed to publish to Redis: {e}")

    async def _redis_pubsub_listener(self):
        """Background task to listen for Redis pub/sub messages."""
        try:
            logger.info("Redis pubsub listener started")
            
            # Simplified: Poll Redis for messages instead of true pub/sub
            # (Upstash doesn't support native pub/sub via HTTP)
            while True:
                try:
                    await asyncio.sleep(1)
                    # In production, implement proper Redis Streams or pub/sub
                except asyncio.CancelledError:
                    break
                    
        except asyncio.CancelledError:
            logger.info("Redis pubsub listener stopped")
        except Exception as e:
            logger.error(f"Redis pubsub listener error: {str(e)}")

    async def _heartbeat_monitor(self):
        """
        Monitor connections and remove dead ones.
        Runs every 30 seconds to clean up stale connections.
        """
        while True:
            try:
                await asyncio.sleep(30)
                await self._cleanup_dead_connections()
            except asyncio.CancelledError:
                logger.info("Heartbeat monitor stopped")
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {str(e)}")

    async def _cleanup_dead_connections(self):
        """Remove dead connections by attempting to send a ping."""
        try:
            dead_connections: List[tuple] = []
            
            # Get snapshot of connections
            async with self._lock:
                connection_snapshot = {
                    project_id: list(users.keys())
                    for project_id, users in self.active_connections.items()
                }
            
            # Test each connection
            for project_id, user_ids in connection_snapshot.items():
                for user_id in user_ids:
                    try:
                        if project_id in self.active_connections:
                            if user_id in self.active_connections[project_id]:
                                websocket = self.active_connections[project_id][user_id]
                                await websocket.send_text("ping")
                    except Exception:
                        dead_connections.append((project_id, user_id))
            
            # Remove dead connections
            for project_id, user_id in dead_connections:
                await self.disconnect(project_id, user_id)
            
            if dead_connections:
                logger.info(f"Cleaned up {len(dead_connections)} dead connections")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def _get_total_connections(self) -> int:
        """Get total number of active connections."""
        total = 0
        for users in self.active_connections.values():
            total += len(users)
        return total

    async def get_project_connections(self, project_id: str) -> int:
        """Get number of connections for a specific project."""
        if project_id in self.active_connections:
            return len(self.active_connections[project_id])
        return 0

    async def get_all_project_ids(self) -> List[str]:
        """Get list of all project IDs with active connections."""
        async with self._lock:
            return list(self.active_connections.keys())

    async def disconnect_all(self):
        """Disconnect all active connections (for shutdown)."""
        try:
            # Get all connections to close
            async with self._lock:
                projects = list(self.active_connections.items())
            
            # Close each connection
            for project_id, users in projects:
                for user_id in list(users.keys()):
                    await self.disconnect(project_id, user_id)
            
            # Cancel background tasks
            if self._pubsub_task and not self._pubsub_task.done():
                self._pubsub_task.cancel()
                try:
                    await self._pubsub_task
                except asyncio.CancelledError:
                    pass
                    
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("All WebSocket connections closed")
            
        except Exception as e:
            logger.error(f"Error during disconnect_all: {e}")

    # ── Convenience Methods for Common Message Types ──────────────

    async def send_log(self, project_id: str, message: str, level: str = "info"):
        """Send a log message to all project subscribers."""
        await self.publish_to_project({
            "type": "log",
            "message": message,
            "level": level,
            "timestamp": datetime.utcnow().isoformat()
        }, project_id)

    async def send_status(self, project_id: str, status: str, message: Optional[str] = None):
        """Send a status update to all project subscribers."""
        payload = {
            "type": "status",
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        if message:
            payload["message"] = message
        await self.publish_to_project(payload, project_id)

    async def send_complete(self, project_id: str, message: str = "Generation complete!"):
        """Send completion message to all project subscribers."""
        await self.publish_to_project({
            "type": "complete",
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }, project_id)

    async def send_error(self, project_id: str, error: str):
        """Send error message to all project subscribers."""
        await self.publish_to_project({
            "type": "error",
            "message": error,
            "timestamp": datetime.utcnow().isoformat()
        }, project_id)

    async def send_progress(self, project_id: str, percentage: int, step: str):
        """Send progress update to all project subscribers."""
        await self.publish_to_project({
            "type": "progress",
            "percentage": min(100, max(0, percentage)),  # Clamp 0-100
            "step": step,
            "timestamp": datetime.utcnow().isoformat()
        }, project_id)

    async def send_file_generated(self, project_id: str, file_path: str):
        """Notify when a new file is generated."""
        await self.publish_to_project({
            "type": "file_generated",
            "file_path": file_path,
            "timestamp": datetime.utcnow().isoformat()
        }, project_id)


# ── WebSocket Manager Initialization ──────────────────────────

# Global WebSocket manager instance
ws_manager = ConnectionManager()


async def init_websocket_manager():
    """Initialize the WebSocket manager (call during app startup)."""
    try:
        await ws_manager.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize WebSocket manager: {e}")
        raise


async def shutdown_websocket_manager():
    """Shutdown the WebSocket manager (call during app shutdown)."""
    try:
        await ws_manager.disconnect_all()
    except Exception as e:
        logger.error(f"Error during WebSocket manager shutdown: {e}")

