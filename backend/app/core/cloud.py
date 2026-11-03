"""
Cloud runtime for serverless hosting (Vercel).

Two problems are solved here so the rest of the Flask app can stay file based:

1. Shared files. Every function instance has its own disk. All files under CLOUD_DATA_ROOT
   are mirrored to a Postgres table. Before each request the instance pulls changes from the
   database, after each request it pushes the files that changed.
2. Realtime events. Serverless has no websockets. `emit()` writes an event row and clients read
   new rows through GET /events/poll (see app/core/events.py).

Required: DATABASE_URL (Neon, added by the Vercel marketplace integration).
"""

import json
import logging
import os
import threading
from typing import Dict, Optional, Tuple

from app.core.paths import CLOUD_DATA_ROOT, IS_CLOUD

logger = logging.getLogger(__name__)

# shortcut: whole files are stored as bytea, fine up to a few hundred MB total on Neon free tier
# (0.5 GB), move large files to object storage when the data set grows beyond that.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS pc_files (
    path    TEXT PRIMARY KEY,
    data    BYTEA,
    size    BIGINT NOT NULL DEFAULT 0,
    deleted BOOLEAN NOT NULL DEFAULT FALSE,
    rev     BIGINT NOT NULL
);
CREATE SEQUENCE IF NOT EXISTS pc_rev_seq;
CREATE TABLE IF NOT EXISTS pc_users (
    id            BIGSERIAL PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS pc_events (
    id      BIGSERIAL PRIMARY KEY,
    name    TEXT NOT NULL,
    payload JSONB NOT NULL,
    sender  TEXT,
    created TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_lock = threading.RLock()
_conn = None
_schema_ready = False
# path -> rev we last pulled or pushed. Used to skip unchanged files.
_known: Dict[str, int] = {}
# path -> (size, mtime_ns) at the last sync. Used to find local changes.
# Directories are stored with a trailing "/" so empty folders survive.
_local_state: Dict[str, Tuple[int, int]] = {}


def enabled() -> bool:
    return IS_CLOUD and bool(_database_url())


def _database_url() -> Optional[str]:
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")


def _connect():
    """Return a live autocommit connection, reconnecting if the previous one dropped."""
    global _conn, _schema_ready
    import psycopg

    with _lock:
        if _conn is not None and not _conn.closed:
            try:
                _conn.execute("SELECT 1")
                return _conn
            except Exception:
                try:
                    _conn.close()
                except Exception:
                    pass
                _conn = None

        url = _database_url()
        if not url:
            raise RuntimeError("DATABASE_URL is not set")
        _conn = psycopg.connect(url, autocommit=True, connect_timeout=10)
        if not _schema_ready:
            _conn.execute(_SCHEMA)
            _schema_ready = True
        return _conn


def _abs(rel_path: str) -> str:
    return os.path.join(CLOUD_DATA_ROOT, *rel_path.rstrip("/").split("/"))


def _signature(full: str) -> Optional[Tuple[int, int]]:
    try:
        st = os.stat(full)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def _scan() -> Dict[str, Tuple[int, int]]:
    state: Dict[str, Tuple[int, int]] = {}
    for root, dirs, files in os.walk(CLOUD_DATA_ROOT):
        for name in dirs:
            rel = os.path.relpath(os.path.join(root, name), CLOUD_DATA_ROOT).replace(os.sep, "/")
            state[rel + "/"] = (0, 0)
        for name in files:
            full = os.path.join(root, name)
            sig = _signature(full)
            if sig is not None:
                state[os.path.relpath(full, CLOUD_DATA_ROOT).replace(os.sep, "/")] = sig
    return state


def pull() -> None:
    """Bring this instance's disk up to date with the database."""
    if not enabled():
        return
    os.makedirs(CLOUD_DATA_ROOT, exist_ok=True)
    with _lock:
        conn = _connect()
        rows = conn.execute("SELECT path, rev, deleted FROM pc_files").fetchall()
        remote = {path: (rev, deleted) for path, rev, deleted in rows}

        # Directories first so files can be written into them.
        stale = [p for p, (rev, deleted) in remote.items() if not deleted and _known.get(p) != rev]
        for path in sorted(stale, key=lambda p: (not p.endswith("/"), p)):
            if path.endswith("/"):
                os.makedirs(_abs(path), exist_ok=True)
                _known[path] = remote[path][0]
                _local_state[path] = (0, 0)
                continue
            row = conn.execute("SELECT data, rev FROM pc_files WHERE path = %s AND NOT deleted", (path,)).fetchone()
            if not row:
                continue
            data, rev = row
            target = _abs(path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as handle:
                handle.write(bytes(data or b""))
            _known[path] = rev
            # Record the signature of what we just wrote so push() does not send it back.
            sig = _signature(target)
            if sig is not None:
                _local_state[path] = sig

        # Entries deleted elsewhere. Deepest paths first so directories are empty when removed.
        gone = [p for p in _known if p not in remote or remote[p][1]]
        for path in sorted(gone, key=len, reverse=True):
            target = _abs(path)
            try:
                os.rmdir(target) if path.endswith("/") else os.remove(target)
            except OSError:
                pass
            _known.pop(path, None)
            _local_state.pop(path, None)


def push() -> None:
    """Write files that changed since the last pull/push to the database."""
    if not enabled():
        return
    with _lock:
        if not os.path.isdir(CLOUD_DATA_ROOT):
            return
        current = _scan()
        changed = [p for p, sig in current.items() if _local_state.get(p) != sig]
        removed = [p for p in _local_state if p not in current]
        if not changed and not removed:
            return

        conn = _connect()
        for path in changed:
            data = b""
            if not path.endswith("/"):
                try:
                    with open(_abs(path), "rb") as handle:
                        data = handle.read()
                except OSError:
                    continue
            _known[path] = conn.execute(
                """
                INSERT INTO pc_files (path, data, size, deleted, rev)
                VALUES (%s, %s, %s, FALSE, nextval('pc_rev_seq'))
                ON CONFLICT (path) DO UPDATE
                SET data = EXCLUDED.data, size = EXCLUDED.size, deleted = FALSE, rev = EXCLUDED.rev
                RETURNING rev
                """,
                (path, data, len(data)),
            ).fetchone()[0]
            _local_state[path] = current[path]

        for path in removed:
            conn.execute(
                """
                INSERT INTO pc_files (path, data, size, deleted, rev)
                VALUES (%s, NULL, 0, TRUE, nextval('pc_rev_seq'))
                ON CONFLICT (path) DO UPDATE SET data = NULL, size = 0, deleted = TRUE, rev = EXCLUDED.rev
                """,
                (path,),
            )
            _known.pop(path, None)
            _local_state.pop(path, None)


def emit(name: str, payload: Optional[dict] = None, sender: Optional[str] = None) -> Optional[int]:
    """Publish an event that polling clients will receive. Pushes pending file changes first."""
    if not enabled():
        return None
    try:
        # Clients react to events by re-reading files, so files must be in the database already.
        push()
        conn = _connect()
        with _lock:
            row = conn.execute(
                "INSERT INTO pc_events (name, payload, sender) VALUES (%s, %s::jsonb, %s) RETURNING id",
                (name, json.dumps(payload or {}, default=str), sender),
            ).fetchone()
            conn.execute("DELETE FROM pc_events WHERE created < now() - interval '15 minutes'")
        return row[0]
    except Exception as exc:
        logger.error("emit(%s) failed: %s", name, exc)
        return None


def poll_events(since: int, limit: int = 200):
    """Return (events, last_id). A negative `since` only returns the current last id."""
    conn = _connect()
    if since < 0:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM pc_events").fetchone()
        return [], int(row[0])
    rows = conn.execute(
        "SELECT id, name, payload, sender FROM pc_events WHERE id > %s ORDER BY id LIMIT %s",
        (since, limit),
    ).fetchall()
    events = [{"id": r[0], "name": r[1], "data": r[2], "sender": r[3]} for r in rows]
    last_id = events[-1]["id"] if events else since
    return events, int(last_id)


def init_app(app) -> None:
    """Attach the sync hooks to a Flask app."""
    from flask import request

    skip_prefixes = ("/events/", "/health", "/ping")

    @app.before_request
    def _cloud_pull():
        if request.method == "OPTIONS" or request.path.startswith(skip_prefixes):
            return None
        try:
            pull()
        except Exception as exc:
            logger.error("cloud pull failed: %s", exc)
        return None

    @app.after_request
    def _cloud_push(response):
        if request.method == "OPTIONS" or request.path.startswith(skip_prefixes):
            return response
        try:
            push()
        except Exception as exc:
            logger.error("cloud push failed: %s", exc)
        return response

    logger.info("[CLOUD] storage mirror enabled (root=%s)", CLOUD_DATA_ROOT)
