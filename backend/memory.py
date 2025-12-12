"""Conversation memory backed by PostgreSQL.

Provides persistent chat history per session for contextual follow-ups.
"""
import logging
from typing import List

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import get_settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Session based conversation memory using PostgreSQL."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._conn = None
        self._ensure_table()
    
    def _get_connection(self):
        """Get database connection."""
        if self._conn is None or self._conn.closed:
            settings = get_settings()
            self._conn = psycopg2.connect(
                dbname=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
                host=settings.postgres_host,
                port=settings.postgres_port,
            )
        return self._conn
    
    def _ensure_table(self):
        """Create memory table if needed."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_memory (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_session 
                    ON conversation_memory(session_id)
                """)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to create memory table: {e}")
    
    def add_message(self, role: str, content: str):
        """Store a message."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversation_memory (session_id, role, content) VALUES (%s, %s, %s)",
                    (self.session_id, role, content)
                )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
    
    def get_history(self, limit: int = 10) -> List[dict]:
        """Get recent messages."""
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT role, content, created_at 
                       FROM conversation_memory
                       WHERE session_id = %s
                       ORDER BY created_at DESC
                       LIMIT %s""",
                    (self.session_id, limit)
                )
                rows = cur.fetchall()
            return list(reversed(rows))
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []
    
    def get_history_text(self, limit: int = 5) -> str:
        """Get history as formatted text for prompts."""
        messages = self.get_history(limit)
        if not messages:
            return ""
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    
    def clear(self):
        """Clear session history."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_memory WHERE session_id = %s",
                    (self.session_id,)
                )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
    
    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
