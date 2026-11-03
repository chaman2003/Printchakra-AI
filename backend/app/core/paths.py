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


def app_data_dir() -> str:
    """The `backend/app/data` directory several route modules use, moved to the cloud root on Vercel."""
    return data_root(os.path.join(APP_DIR, "data"))
