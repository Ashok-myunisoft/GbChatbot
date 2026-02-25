"""
db_setup.py — PostgreSQL connection and schema initialisation.
Import `get_pg_conn` wherever a DB connection is needed.
Run this file directly to create/verify tables:
    python db_setup.py
"""

import os
import logging
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

POSTGRES_URL = os.environ.get("POSTGRES_URL")


def get_pg_conn():
    """Return a new psycopg2 connection to the PostgreSQL database."""
    return psycopg2.connect(POSTGRES_URL)


def create_tables():
    """
    Create all required application tables if they don't already exist.
    Safe to call on every startup (idempotent).
    """
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:

            # ------------------------------------------------------------------
            # Table 1: conversation_threads
            # Replaces Firestore 'conversation_threads' collection.
            # Each row is one chat thread; messages stored as a JSONB array.
            # ------------------------------------------------------------------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_threads (
                    thread_id  TEXT PRIMARY KEY,
                    username   TEXT NOT NULL,
                    title      TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    messages   JSONB   DEFAULT '[]',
                    is_active  BOOLEAN DEFAULT TRUE,
                    user_role  TEXT,
                    user_name  TEXT
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_threads_username
                    ON conversation_threads(username);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_threads_updated
                    ON conversation_threads(updated_at DESC);
            """)

            # ------------------------------------------------------------------
            # Table 2: memory_vectors
            # Replaces GCS FAISS blob storage.
            # Each row is one conversation turn; FAISS is rebuilt in-memory
            # from these rows on startup.
            # ------------------------------------------------------------------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    id           SERIAL PRIMARY KEY,
                    memory_id    TEXT UNIQUE,
                    username     TEXT NOT NULL,
                    user_role    TEXT,
                    bot_type     TEXT,
                    thread_id    TEXT,
                    content      TEXT NOT NULL,
                    user_message TEXT,
                    bot_response TEXT,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mv_username
                    ON memory_vectors(username);
            """)

            # ------------------------------------------------------------------
            # Table 3: user_sessions
            # Replaces Firestore 'user_sessions' collection.
            # Tracks per-user activity and role.
            # ------------------------------------------------------------------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    username            TEXT PRIMARY KEY,
                    first_seen          TEXT,
                    last_activity       TEXT,
                    session_count       INTEGER DEFAULT 1,
                    total_interactions  INTEGER DEFAULT 1,
                    name                TEXT,
                    user_role           TEXT
                );
            """)

        conn.commit()
        logger.info("✅ PostgreSQL tables ready.")
        print("✅ PostgreSQL tables created / verified successfully.")

    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


# Allow running directly to set up the database
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_tables()
