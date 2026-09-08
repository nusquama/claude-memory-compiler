"""Durable, stdlib-only ACE outbox.

The outbox is the boundary between local transcript parsing and a remote
worker.  Envelopes are inserted transactionally before a source cursor may be
advanced by the caller.  The idempotency tuple is
``(source, project.id, session_id, revision)``; retries never rewrite an
already acknowledged payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


SCHEMA_VERSION = 1
DEFAULT_LEASE_SECONDS = 300.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OutboxError(RuntimeError):
    """Base class for durable outbox errors."""


class InvalidEnvelopeError(OutboxError):
    """The payload is not a valid shared ACE envelope."""


class PayloadTooLargeError(OutboxError):
    """A payload or lot exceeds its configured byte bound."""


class OutboxNotFoundError(OutboxError):
    """The caller referenced a key that is not in the outbox."""


class OutboxKey(str):
    """String key that also works with callers expecting ``result.key``."""

    @property
    def key(self) -> str:
        return str(self)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> float:
    return time.time()


def _safe_error(value: Any) -> str:
    text = str(value or "error")
    # Keep retry diagnostics useful while avoiding accidental credential echo
    # if a connector includes a response fragment in its exception.
    try:
        from utils import redact_sensitive_text

        text = redact_sensitive_text(text)
    except Exception:
        text = re.sub(
            r"(?i)(api[_-]?key|token|password|secret|cookie)\s*[:=]\s*[^\s,;]+",
            r"\1=<REDACTED>",
            text,
        )
    return text[:4000]


def _identity(envelope: Mapping[str, Any]) -> tuple[str, str, str, str]:
    project = envelope.get("project")
    if not isinstance(project, Mapping):
        raise InvalidEnvelopeError("project must be an object")
    project_id = str(project.get("id") or project.get("project_id") or "")
    return (
        str(envelope.get("source") or ""),
        project_id,
        str(envelope.get("session_id") or ""),
        str(envelope.get("revision") or ""),
    )


def outbox_key(envelope: Mapping[str, Any]) -> str:
    """Return the stable idempotency key for an envelope."""

    source, project_id, session_id, revision = _identity(envelope)
    return hashlib.sha256(
        _canonical(
            {
                "source": source,
                "project_id": project_id,
                "session_id": session_id,
                "revision": revision,
            }
        ).encode("utf-8")
    ).hexdigest()


def _validate_envelope(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, Mapping):
        raise InvalidEnvelopeError("envelope must be an object")
    payload = dict(envelope)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InvalidEnvelopeError("unsupported envelope schema_version")
    source = payload.get("source")
    if source not in {"codex", "claude", "hermes"}:
        raise InvalidEnvelopeError("unsupported envelope source")
    project = payload.get("project")
    if not isinstance(project, Mapping):
        raise InvalidEnvelopeError("project must be an object")
    for key in ("id", "name", "root", "vault_dir"):
        if project.get(key) in (None, ""):
            raise InvalidEnvelopeError(f"project.{key} is required")
    for key in ("session_id", "revision", "source_path"):
        if payload.get(key) in (None, ""):
            raise InvalidEnvelopeError(f"{key} is required")
    revision = str(payload["revision"])
    if not _SHA256.fullmatch(revision):
        raise InvalidEnvelopeError("revision must be a SHA-256 hex digest")
    for key in ("messages", "attachments"):
        if not isinstance(payload.get(key), list):
            raise InvalidEnvelopeError(f"{key} must be a list")
    try:
        _canonical(payload)
    except (TypeError, ValueError) as exc:
        raise InvalidEnvelopeError("envelope is not JSON serializable") from exc
    return payload


@dataclass(frozen=True)
class PendingRecord:
    key: str
    envelope: dict[str, Any]
    attempts: int
    status: str
    created_at: float
    updated_at: float
    payload_bytes: int
    last_error: str | None = None

    @property
    def payload(self) -> dict[str, Any]:
        """Alias used by connector workers that call the envelope payload."""

        return self.envelope

    def __getitem__(self, key: str) -> Any:
        if key == "key":
            return self.key
        if key == "envelope":
            return self.envelope
        if key == "payload":
            return self.envelope
        if key == "attempts":
            return self.attempts
        if key == "status":
            return self.status
        if key == "payload_bytes":
            return self.payload_bytes
        if key == "last_error":
            return self.last_error
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "envelope": self.envelope,
            "attempts": self.attempts,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "payload_bytes": self.payload_bytes,
            "last_error": self.last_error,
        }


class Outbox:
    """SQLite-backed transactional queue with bounded claims and retries."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_payload_bytes: int = 4_000_000,
        max_lot_bytes: int = 8_000_000,
        max_lot_items: int = 50,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> None:
        self.db_path = str(db_path)
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_lot_bytes = int(max_lot_bytes)
        self.max_lot_items = int(max_lot_items)
        self.lease_seconds = float(lease_seconds)
        if self.max_payload_bytes <= 0 or self.max_lot_bytes <= 0 or self.max_lot_items <= 0:
            raise ValueError("outbox limits must be positive")
        path = Path(self.db_path)
        if self.db_path != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ace_outbox (
                key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                project_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                revision TEXT NOT NULL,
                payload TEXT NOT NULL,
                payload_bytes INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','inflight','retry','acknowledged')),
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL DEFAULT 0,
                lease_until REAL,
                receipt TEXT,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(source, project_id, session_id, revision)
            );
            CREATE INDEX IF NOT EXISTS idx_ace_outbox_available
                ON ace_outbox(status, next_attempt_at, created_at);
            CREATE TABLE IF NOT EXISTS ace_outbox_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO ace_outbox_meta(key, value) VALUES ('dispatch_cursor', '0');
            """
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Outbox":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def enqueue(self, envelope: Mapping[str, Any]) -> OutboxKey:
        """Durably enqueue a filtered envelope and return its idempotency key."""

        payload = _validate_envelope(envelope)
        serialized = _canonical(payload)
        payload_bytes = len(serialized.encode("utf-8"))
        if payload_bytes > self.max_payload_bytes:
            raise PayloadTooLargeError(
                f"payload is {payload_bytes} bytes; limit is {self.max_payload_bytes}"
            )
        source, project_id, session_id, revision = _identity(payload)
        if not all((source, project_id, session_id, revision)):
            raise InvalidEnvelopeError("source/project/session/revision identity is incomplete")
        key = outbox_key(payload)
        now = _now()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                """
                INSERT INTO ace_outbox(
                    key, source, project_id, session_id, revision, payload,
                    payload_bytes, status, attempts, next_attempt_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, 0, ?, ?)
                ON CONFLICT(source, project_id, session_id, revision) DO NOTHING
                """,
                (
                    key,
                    source,
                    project_id,
                    session_id,
                    revision,
                    serialized,
                    payload_bytes,
                    now,
                    now,
                ),
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return OutboxKey(key)

    def _available_rows(
        self,
        now: float,
        project_id: str | None = None,
        created_after: float | None = None,
    ) -> list[sqlite3.Row]:
        # An interrupted sender must not strand a claim forever.  Expired
        # leases are made retryable in the same transaction as the next claim.
        if project_id is None:
            self.connection.execute(
                """
                UPDATE ace_outbox
                   SET status='retry', lease_until=NULL, next_attempt_at=?, updated_at=?
                 WHERE status='inflight' AND lease_until IS NOT NULL AND lease_until <= ?
                """,
                (now, now, now),
            )
            if created_after is None:
                query = """
                    SELECT * FROM ace_outbox
                     WHERE status IN ('pending','retry') AND next_attempt_at <= ?
                     ORDER BY created_at DESC, key DESC
                """
                params = (now,)
            else:
                query = """
                    SELECT * FROM ace_outbox
                     WHERE status IN ('pending','retry') AND next_attempt_at <= ?
                       AND created_at >= ?
                     ORDER BY created_at DESC, key DESC
                """
                params = (now, float(created_after))
            return list(self.connection.execute(query, params))
        scoped_project = str(project_id)
        self.connection.execute(
            """
            UPDATE ace_outbox
               SET status='retry', lease_until=NULL, next_attempt_at=?, updated_at=?
             WHERE project_id=? AND status='inflight'
               AND lease_until IS NOT NULL AND lease_until <= ?
            """,
            (now, now, scoped_project, now),
        )
        if created_after is None:
            query = """
                SELECT * FROM ace_outbox
                 WHERE project_id=? AND status IN ('pending','retry')
                   AND next_attempt_at <= ?
                 ORDER BY created_at DESC, key DESC
            """
            params = (scoped_project, now)
        else:
            query = """
                SELECT * FROM ace_outbox
                 WHERE project_id=? AND status IN ('pending','retry')
                   AND next_attempt_at <= ? AND created_at >= ?
                 ORDER BY created_at DESC, key DESC
            """
            params = (scoped_project, now, float(created_after))
        return list(self.connection.execute(query, params))

    def _dispatch_cursor(self) -> tuple[float, str] | None:
        """Read the last dispatch position without trusting stale metadata.

        Older ACE outboxes stored an integer round-robin counter here.  Such a
        value is intentionally treated as an absent cursor and replaced on
        the next successful claim; it must never affect which rows are
        eligible.  The current cursor is a stable ``(created_at, key)``
        marker so newly inserted rows cannot move the fairness window.
        """

        row = self.connection.execute(
            "SELECT value FROM ace_outbox_meta WHERE key='dispatch_cursor'"
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(str(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(value, Mapping):
            return None
        created_at = value.get("created_at")
        key = value.get("key")
        if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
            return None
        if not isinstance(key, str) or not key:
            return None
        return float(created_at), key

    def _set_dispatch_cursor(self, row: sqlite3.Row) -> None:
        """Persist a stable marker after the rows selected for this claim."""

        marker = _canonical(
            {"created_at": float(row["created_at"]), "key": str(row["key"])}
        )
        self.connection.execute(
            "UPDATE ace_outbox_meta SET value=? WHERE key='dispatch_cursor'",
            (marker,),
        )

    def _fair_select(self, rows: list[sqlite3.Row], limit: int) -> list[sqlite3.Row]:
        """Select a bounded rotating window over the available rows.

        ``_available_rows`` is ordered newest-first for compatibility with the
        connector.  Reserving only the oldest row leaves every middle row
        behind when new rows keep arriving.  A stable marker instead advances
        through the complete ordered ring: after a claim, the next window
        starts at the first row older than the marker and wraps to the newest
        row at the end.  New arrivals therefore join the next cycle while
        existing rows retain a finite wait bound of one ring traversal.
        """

        if not rows or limit <= 0:
            return []
        count = min(len(rows), limit)
        cursor = self._dispatch_cursor()
        start = 0
        if cursor is not None:
            for index, row in enumerate(rows):
                position = (float(row["created_at"]), str(row["key"]))
                # Rows are newest-first, so a strictly older row advances the
                # marker.  Equal timestamps remain totally ordered by key.
                if position < cursor:
                    start = index
                    break
            else:
                # The marker is at or older than every available row.  This
                # is the end-of-ring case, so wrap to the newest row.
                start = 0
        return [rows[(start + offset) % len(rows)] for offset in range(count)]

    def pending(
        self,
        limit: int | None = None,
        project_id: str | None = None,
        created_after: float | None = None,
    ) -> list[PendingRecord]:
        """Claim an available lot, optionally scoped to one project.

        The project predicate is applied inside the SQLite claim transaction,
        before any row becomes ``inflight``.  This is required by the native
        multi-project tick: filtering after ``pending()`` would already have
        leased another project's rows.
        """

        requested = self.max_lot_items if limit is None else int(limit)
        if requested <= 0:
            return []
        requested = min(requested, self.max_lot_items)
        now = _now()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self._available_rows(
                now,
                project_id=project_id,
                created_after=created_after,
            )
            selected = self._fair_select(rows, requested)
            total_bytes = sum(int(row["payload_bytes"]) for row in selected)
            if total_bytes > self.max_lot_bytes:
                # Keep the claim bounded even when individual payloads are
                # valid.  Newest/backlog fairness applies to the rows that fit.
                bounded: list[sqlite3.Row] = []
                used = 0
                for row in selected:
                    size = int(row["payload_bytes"])
                    if bounded and used + size > self.max_lot_bytes:
                        continue
                    if not bounded and size > self.max_lot_bytes:
                        raise PayloadTooLargeError("a pending lot item exceeds max_lot_bytes")
                    bounded.append(row)
                    used += size
                selected = bounded
            if selected:
                # Advance only after byte bounds have been applied.  A row
                # that remains pending because it cannot fit must not move the
                # ring marker and disappear from the next fair window.
                self._set_dispatch_cursor(selected[-1])
            lease_until = now + self.lease_seconds
            for row in selected:
                self.connection.execute(
                    """
                    UPDATE ace_outbox
                       SET status='inflight', attempts=attempts+1,
                           lease_until=?, updated_at=?
                     WHERE key=? AND status IN ('pending','retry')
                    """,
                    (lease_until, now, row["key"]),
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

        result: list[PendingRecord] = []
        for row in selected:
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError as exc:
                raise OutboxError(f"corrupt payload for outbox key {row['key']}") from exc
            result.append(
                PendingRecord(
                    key=str(row["key"]),
                    envelope=payload,
                    attempts=int(row["attempts"]) + 1,
                    status="inflight",
                    created_at=float(row["created_at"]),
                    updated_at=now,
                    payload_bytes=int(row["payload_bytes"]),
                    last_error=row["last_error"],
                )
            )
        return result

    claim = pending

    def snapshots(
        self, project_id: str | None = None, limit: int | None = None
    ) -> list[PendingRecord]:
        """Read envelopes for local extraction without claiming or leasing.

        Every status is eligible: the local memory path must not depend on
        the remote acknowledgement.  Rows are returned oldest first so a
        session's revisions are replayed in order.
        """

        sql = "SELECT * FROM ace_outbox"
        params: list[Any] = []
        if project_id is not None:
            sql += " WHERE project_id = ?"
            params.append(str(project_id))
        sql += " ORDER BY created_at ASC, key ASC"
        if limit is not None and int(limit) > 0:
            sql += " LIMIT ?"
            params.append(int(limit))
        result: list[PendingRecord] = []
        for row in self.connection.execute(sql, params):
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError as exc:
                raise OutboxError(f"corrupt payload for outbox key {row['key']}") from exc
            result.append(
                PendingRecord(
                    key=str(row["key"]),
                    envelope=payload,
                    attempts=int(row["attempts"]),
                    status=str(row["status"]),
                    created_at=float(row["created_at"]),
                    updated_at=float(row["updated_at"]),
                    payload_bytes=int(row["payload_bytes"]),
                    last_error=row["last_error"],
                )
            )
        return result

    def iter_pending(
        self, limit: int | None = None, project_id: str | None = None
    ) -> Iterator[PendingRecord]:
        yield from self.pending(limit, project_id=project_id)

    def ack(self, key: str, receipt: Any) -> bool:
        """Acknowledge only with a non-empty confirmed connector receipt."""

        if receipt is None or receipt is False or receipt == "" or receipt == {}:
            raise OutboxError("acknowledgement requires a confirmed database receipt")
        serialized_receipt = _canonical(receipt)
        now = _now()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT status, receipt FROM ace_outbox WHERE key=?", (str(key),)
            ).fetchone()
            if row is None:
                raise OutboxNotFoundError(str(key))
            if row["status"] == "acknowledged":
                if row["receipt"] == serialized_receipt:
                    self.connection.execute("COMMIT")
                    return False
                raise OutboxError("outbox key already acknowledged with another receipt")
            self.connection.execute(
                """
                UPDATE ace_outbox
                   SET status='acknowledged', receipt=?, lease_until=NULL,
                       updated_at=?, last_error=NULL
                 WHERE key=?
                """,
                (serialized_receipt, now, str(key)),
            )
            self.connection.execute("COMMIT")
            return True
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def fail(self, key: str, error: Any, *, retry_at: float | None = None) -> bool:
        """Return a claimed item to the retry queue without deleting it."""

        now = _now()
        next_attempt = now if retry_at is None else float(retry_at)
        message = _safe_error(error)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT status FROM ace_outbox WHERE key=?", (str(key),)
            ).fetchone()
            if row is None:
                raise OutboxNotFoundError(str(key))
            if row["status"] == "acknowledged":
                raise OutboxError("cannot retry an acknowledged outbox item")
            self.connection.execute(
                """
                UPDATE ace_outbox
                   SET status='retry', next_attempt_at=?, lease_until=NULL,
                       updated_at=?, last_error=?
                 WHERE key=?
                """,
                (next_attempt, now, message, str(key)),
            )
            self.connection.execute("COMMIT")
            return True
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    retry = fail

    def summary(self) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) AS count, COALESCE(SUM(payload_bytes),0) AS bytes "
            "FROM ace_outbox GROUP BY status"
        ).fetchall()
        statuses = {
            status: {"count": int(row["count"]), "bytes": int(row["bytes"])}
            for row in rows
            for status in [str(row["status"])]
        }
        total = sum(item["count"] for item in statuses.values())
        return {"total": total, "statuses": statuses}

    stats = summary


def _cli() -> int:
    parser = argparse.ArgumentParser(description="ACE durable outbox")
    parser.add_argument("--db", help="SQLite path (only used by the optional status command)")
    subparsers = parser.add_subparsers(dest="command")
    status_parser = subparsers.add_parser("status", help="show queue counters")
    status_parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if args.command == "status":
        db = args.path or args.db
        if not db:
            parser.error("status requires a SQLite path")
        with Outbox(db) as queue:
            print(json.dumps(queue.summary(), sort_keys=True))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI smoke/help only
    raise SystemExit(_cli())
