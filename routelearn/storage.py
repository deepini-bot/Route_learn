"""SQLite-backed shared learning state and request guardrails."""
from __future__ import annotations

import io
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

import numpy as np


class QuotaExceeded(Exception):
    def __init__(self, quota: str):
        self.quota = quota
        super().__init__(quota)


class SQLiteStateStore:
    def __init__(self, path: str):
        if path == ":memory:":
            raise ValueError("SQLiteStateStore requires a file-backed database")
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS policy_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_feedback (
                    request_id TEXT PRIMARY KEY,
                    arm INTEGER NOT NULL,
                    features BLOB NOT NULL,
                    identity TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quota_usage (
                    reservation_id TEXT PRIMARY KEY,
                    identity TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    usage_date TEXT NOT NULL,
                    reserved_cost REAL NOT NULL,
                    actual_cost REAL
                );
                CREATE INDEX IF NOT EXISTS quota_usage_identity_time
                    ON quota_usage(identity, created_at);
                CREATE INDEX IF NOT EXISTS quota_usage_identity_date
                    ON quota_usage(identity, usage_date);
                """
            )
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(pending_feedback)")
            }
            if "identity" not in columns:
                db.execute(
                    "ALTER TABLE pending_feedback ADD COLUMN identity TEXT NOT NULL "
                    "DEFAULT 'shared'"
                )
            if "created_at" not in columns:
                db.execute(
                    "ALTER TABLE pending_feedback ADD COLUMN created_at REAL NOT NULL DEFAULT 0"
                )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout = 30000")
        db.execute("PRAGMA journal_mode = WAL")
        return db

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _policy_payload(policy) -> bytes:
        output = io.BytesIO()
        np.savez(
            output,
            **policy.model.state(),
            lam=np.asarray(policy.lam),
        )
        return output.getvalue()

    @staticmethod
    def _load_policy_payload(policy, payload: bytes) -> None:
        with np.load(io.BytesIO(payload), allow_pickle=False) as state:
            policy.model.load_state(
                {"A": state["A"], "b": state["b"], "counts": state["counts"]}
            )
            policy.lam = float(state["lam"])

    def initialize_policy(self, policy) -> None:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM policy_state WHERE id = 1").fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO policy_state(id, payload) VALUES(1, ?)",
                    (self._policy_payload(policy),),
                )
            else:
                self._load_policy_payload(policy, row[0])

    def refresh_policy(self, policy) -> None:
        with self._connection() as db:
            row = db.execute("SELECT payload FROM policy_state WHERE id = 1").fetchone()
            if row is not None:
                self._load_policy_payload(policy, row[0])

    def save_policy(self, policy) -> None:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO policy_state(id, payload) VALUES(1, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                (self._policy_payload(policy),),
            )

    def set_lambda(self, policy, lam: float) -> None:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM policy_state WHERE id = 1").fetchone()
            if row is not None:
                self._load_policy_payload(policy, row[0])
            policy.lam = lam
            db.execute(
                "INSERT INTO policy_state(id, payload) VALUES(1, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                (self._policy_payload(policy),),
            )

    def update_policy(self, policy, arm: int, features: np.ndarray, quality: float) -> None:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM policy_state WHERE id = 1").fetchone()
            if row is not None:
                self._load_policy_payload(policy, row[0])
            policy.model.update(arm, features, quality)
            db.execute(
                "INSERT INTO policy_state(id, payload) VALUES(1, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                (self._policy_payload(policy),),
            )

    def save_pending(self, request_id: str, arm: int, features: np.ndarray,
                     identity: str) -> None:
        output = io.BytesIO()
        np.save(output, features)
        with self._connection() as db:
            db.execute(
                "INSERT OR REPLACE INTO pending_feedback"
                "(request_id, arm, features, identity, created_at) VALUES(?, ?, ?, ?, ?)",
                (request_id, arm, output.getvalue(), identity, datetime.now(timezone.utc).timestamp()),
            )
            db.execute(
                "DELETE FROM pending_feedback WHERE request_id IN ("
                "SELECT request_id FROM pending_feedback ORDER BY created_at DESC "
                "LIMIT -1 OFFSET 10000)"
            )

    def consume_pending(self, request_id: str, identity: str) -> tuple[int, np.ndarray] | None:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT arm, features FROM pending_feedback "
                "WHERE request_id = ? AND identity = ?",
                (request_id, identity),
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "DELETE FROM pending_feedback WHERE request_id = ? AND identity = ?",
                (request_id, identity),
            )
        return row[0], np.load(io.BytesIO(row[1]), allow_pickle=False)

    def reserve_usage(self, identity: str, requests_per_minute: int, daily_spend: float,
                      estimated_cost: float, now: float) -> str:
        usage_date = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
        reservation_id = uuid.uuid4().hex
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE quota_usage SET actual_cost = 0 "
                "WHERE actual_cost IS NULL AND created_at < ?",
                (now - 3600,),
            )
            db.execute(
                "DELETE FROM quota_usage WHERE created_at < ?",
                (now - 30 * 86400,),
            )
            if requests_per_minute > 0:
                count = db.execute(
                    "SELECT COUNT(*) FROM quota_usage "
                    "WHERE identity = ? AND created_at > ?",
                    (identity, now - 60),
                ).fetchone()[0]
                if count >= requests_per_minute:
                    raise QuotaExceeded("rate_limit")
            if daily_spend > 0:
                spent = db.execute(
                    "SELECT COALESCE(SUM(COALESCE(actual_cost, reserved_cost)), 0) "
                    "FROM quota_usage WHERE identity = ? AND usage_date = ?",
                    (identity, usage_date),
                ).fetchone()[0]
                if spent + estimated_cost > daily_spend:
                    raise QuotaExceeded("daily_spend")
            db.execute(
                "INSERT INTO quota_usage(reservation_id, identity, created_at, usage_date, "
                "reserved_cost) VALUES(?, ?, ?, ?, ?)",
                (reservation_id, identity, now, usage_date, estimated_cost),
            )
        return reservation_id

    def settle_usage(self, reservation_id: str, actual_cost: float) -> None:
        with self._connection() as db:
            cursor = db.execute(
                "UPDATE quota_usage SET actual_cost = ? WHERE reservation_id = ?",
                (actual_cost, reservation_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Quota reservation was not found")

    def release_usage(self, reservation_id: str) -> None:
        with self._connection() as db:
            db.execute("DELETE FROM quota_usage WHERE reservation_id = ?", (reservation_id,))
