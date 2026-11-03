"""
Where the app keeps files.

Locally the data lives next to the code. On Vercel the code directory is read-only and each
function instance has its own disk, so the data root moves to a temp directory that
`app.core.cloud` mirrors to Postgres.

Set PC_CLOUD=1 to run the cloud code path on a development machine.
"""

import os
import tempfile

IS_CLOUD = bool(os.environ.get("VERCEL") or os.environ.get("PC_CLOUD"))

# Only used when IS_CLOUD is true. Everything under this directory is mirrored to the database.
CLOUD_DATA_ROOT = os.environ.get("PC_DATA_DIR") or os.path.join(tempfile.gettempdir(), "pc-data")


def data_root(local_default: str) -> str:
    """Return the data directory: the cloud temp root on Vercel, otherwise `local_default`."""
    return CLOUD_DATA_ROOT if IS_CLOUD else str(local_default)


def logs_root(local_default: str) -> str:
    """Writable logs directory (the code directory is read-only on Vercel)."""
    return os.path.join(tempfile.gettempdir(), "pc-logs") if IS_CLOUD else str(local_default)


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def current_user_id():
    """Id of the logged-in user of the current request, or None outside a request / when not logged in."""
    from flask import g, has_request_context

    return g.get("user_id") if has_request_context() else None


def user_scope() -> str:
    """Path prefix (relative to the data root, trailing slash) that holds the current user's files."""
    uid = current_user_id()
    return f"users/{int(uid)}/" if uid else ""


def user_root(base: str) -> str:
    """`base` itself without a logged-in user, otherwise that user's private folder below it.

    Every account gets its own folder, so one account can never list, read or overwrite another's files.
    """
    scope = user_scope()
    return os.path.join(str(base), *scope.rstrip("/").split("/")) if scope else str(base)


def app_data_dir() -> str:
    """The `backend/app/data` directory several route modules use, moved to the cloud root on Vercel.

    Call it per request: the result depends on who is logged in."""
    return user_root(data_root(os.path.join(APP_DIR, "data")))
