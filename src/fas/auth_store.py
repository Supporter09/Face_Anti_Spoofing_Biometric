from __future__ import annotations

import os

import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection


def _get_connection() -> PgConnection:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(database_url)


def _init_db() -> None:
    """Create the face_templates table if it doesn't exist."""
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_templates (
                    user_id     TEXT PRIMARY KEY,
                    embedding   DOUBLE PRECISION[],
                    image_base64 TEXT,
                    enrolled_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Add column if it doesn't exist (for existing databases)
            cur.execute("""
                ALTER TABLE face_templates
                ADD COLUMN IF NOT EXISTS image_base64 TEXT
            """)
        conn.commit()


# Run once on import — creates table if missing
_init_db()


class FaceTemplateStore:
    def save_template(self, user_id: str, embedding: np.ndarray) -> None:
        embedding_list = embedding.astype(float).tolist()
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO face_templates (user_id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id) DO UPDATE
                        SET embedding   = EXCLUDED.embedding,
                            enrolled_at = NOW()
                """, (user_id, embedding_list))
            conn.commit()

    def get_template(self, user_id: str) -> np.ndarray | None:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT embedding FROM face_templates WHERE user_id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return np.asarray(row[0], dtype=np.float32)

    def get_all_templates(self) -> dict[str, np.ndarray]:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, embedding FROM face_templates")
                rows = cur.fetchall()
        return {
            user_id: np.asarray(embedding, dtype=np.float32)
            for user_id, embedding in rows
        }

    def user_exists(self, user_id: str) -> bool:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM face_templates WHERE user_id = %s",
                    (user_id,),
                )
                return cur.fetchone() is not None

    def delete_template(self, user_id: str) -> bool:
        """Remove a user's template. Returns True if a row was deleted."""
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM face_templates WHERE user_id = %s",
                    (user_id,),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted