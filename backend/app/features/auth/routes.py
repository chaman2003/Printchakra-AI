"""
Accounts: sign up, log in, and protect the API.

Passwords are hashed with werkzeug (scrypt). A login returns a signed token (7 days) that the
frontend sends as `Authorization: Bearer <token>`. Users live in Postgres when DATABASE_URL is set
(cloud) and in a JSON file otherwise (local development).

Isolation: the token carries the user id; `require_login` puts it in `g.user_id` and every file path
and event is scoped to it (see app.core.paths.user_root and app.core.cloud).
Roles: "user" (default) and "admin". An admin can list accounts and their storage use (GET /admin/users),
but has no access to other users' files.

shortcut: there is no login rate limit beyond the platform's own, add one when the app is public.
"""

import json
import logging
import os
import re
import threading

from flask import Blueprint, g, jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

from app.core import cloud
from app.core.paths import IS_CLOUD

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

TOKEN_MAX_AGE_S = 7 * 24 * 3600
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,23}$")
MIN_PASSWORD = 6

_FIELDS = ("id", "email", "name", "username", "role", "password_hash")

# Accounts created on first use so the deployed app can be tried straight away.
# Disable with SEED_TEST_USER=0. (username, email, name, password, role)
SEED_USERS = (
    ("admin", "admin@printchakra.local", "Administrator", "admin123", "admin"),
    ("test", "test@gmail.com", "Test User", "test123", "user"),
)

_file_lock = threading.Lock()
_seeded = False


# ---------------------------------------------------------------- storage

def _users_file() -> str:
    from app.core.config import get_config

    # The account list is shared by everyone, so it lives in the data root, never in a user's folder.
    return os.path.join(get_config().DATA_DIR, "users.json")


def _read_users() -> dict:
    try:
        with open(_users_file(), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _write_users(users: dict) -> None:
    path = _users_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(users, handle)


def _find_user(identifier: str):
    """Look a user up by email (contains "@") or by username. Returns a dict or None."""
    ident = (identifier or "").strip().lower()
    if not ident:
        return None
    by_email = "@" in ident
    if cloud.enabled():
        column = "email" if by_email else "lower(username)"  # constant strings, never user input
        row = cloud._connect().execute(
            f"SELECT {', '.join(_FIELDS)} FROM pc_users WHERE {column} = %s", (ident,)
        ).fetchone()
        return dict(zip(_FIELDS, row)) if row else None
    with _file_lock:
        users = _read_users()
    found = users.get(ident) if by_email else next(
        (u for u in users.values() if (u.get("username") or "").lower() == ident), None
    )
    return {"username": None, "role": "user", **found} if found else None


def _create_user(email: str, name: str, password: str, username=None, role: str = "user"):
    """Create a user. Returns the user dict, or None when the email or username is already taken."""
    password_hash = generate_password_hash(password)
    if cloud.enabled():
        row = cloud._connect().execute(
            "INSERT INTO pc_users (email, name, password_hash, username, role) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING RETURNING id",
            (email, name, password_hash, username, role),
        ).fetchone()
        return {"id": row[0], "email": email, "name": name, "username": username, "role": role} if row else None
    with _file_lock:
        users = _read_users()
        if email in users or (username and any((u.get("username") or "").lower() == username for u in users.values())):
            return None
        user = {"id": len(users) + 1, "email": email, "name": name, "username": username, "role": role,
                "password_hash": password_hash}
        users[email] = user
        _write_users(users)
        return {k: v for k, v in user.items() if k != "password_hash"}


def _set_username(email: str, username: str, role: str) -> None:
    """Give an account created before usernames existed its username and role."""
    if cloud.enabled():
        cloud._connect().execute(
            "UPDATE pc_users SET username = %s, role = %s WHERE email = %s AND username IS NULL",
            (username, role, email),
        )
        return
    with _file_lock:
        users = _read_users()
        if email in users and not users[email].get("username"):
            users[email].update(username=username, role=role)
            _write_users(users)


def _ensure_seed_users() -> None:
    global _seeded
    if _seeded or os.environ.get("SEED_TEST_USER", "1") == "0":
        return
    try:
        for username, email, name, password, role in SEED_USERS:
            if _find_user(username):
                continue
            if _find_user(email):
                _set_username(email, username, role)
            else:
                _create_user(email, name, password, username, role)
        _seeded = True
    except Exception as exc:
        logger.error("Could not seed the demo accounts: %s", exc)


# ---------------------------------------------------------------- tokens

def _serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        if IS_CLOUD:
            raise RuntimeError("SECRET_KEY must be set")
        secret = "dev-secret-key-change-in-production"
    return URLSafeTimedSerializer(secret, salt="printchakra-auth")


def _issue_token(user: dict) -> str:
    return _serializer().dumps({"uid": user["id"], "email": user["email"]})


def _read_token(token: str):
    try:
        return _serializer().loads(token, max_age=TOKEN_MAX_AGE_S)
    except (BadSignature, SignatureExpired):
        return None


def _token_from_request() -> str:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    # Images and downloads opened by the browser cannot send headers.
    return request.args.get("access_token", "")


def _public(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "username": user.get("username"),
        "role": user.get("role") or "user",
    }


# ---------------------------------------------------------------- guard

# Paths reachable without logging in.
_OPEN_PREFIXES = ("/auth/", "/health", "/ping")


def require_login():
    """before_request hook: reject API calls that carry no valid token, otherwise remember who is calling."""
    if request.method == "OPTIONS" or request.path == "/" or request.path.startswith(_OPEN_PREFIXES):
        return None
    claims = _read_token(_token_from_request())
    if not claims:
        return jsonify({"success": False, "error": "Please log in", "code": "auth_required"}), 401
    g.user_id = int(claims["uid"])
    g.user_email = claims["email"]
    return None


# ---------------------------------------------------------------- routes

def _body() -> dict:
    return request.get_json(silent=True) or {}


@auth_bp.route("/signup", methods=["POST"])
def signup():
    _ensure_seed_users()
    data = _body()
    name = str(data.get("name") or "").strip()[:80]
    username = str(data.get("username") or "").strip().lower()
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")

    if not name:
        return jsonify({"success": False, "error": "Please enter your name"}), 400
    if not USERNAME_RE.match(username):
        return jsonify({"success": False, "error": "Username must be 3-24 letters, numbers, dots, dashes or underscores"}), 400
    if not EMAIL_RE.match(email) or len(email) > 254:
        return jsonify({"success": False, "error": "Please enter a valid email address"}), 400
    if len(password) < MIN_PASSWORD or len(password) > 200:
        return jsonify({"success": False, "error": f"Password must be at least {MIN_PASSWORD} characters"}), 400

    if _find_user(username):
        return jsonify({"success": False, "error": "This username is already taken"}), 409
    user = _create_user(email, name, password, username)
    if user is None:
        return jsonify({"success": False, "error": "An account with this email already exists"}), 409
    return jsonify({"success": True, "token": _issue_token(user), "user": _public(user)}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    _ensure_seed_users()
    data = _body()
    # "identifier" is a username or an email. "email" is still accepted for older clients.
    identifier = str(data.get("identifier") or data.get("username") or data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")

    user = _find_user(identifier)
    # Always run one hash check so an unknown account takes as long as a wrong password.
    valid = check_password_hash(user["password_hash"] if user else generate_password_hash("x"), password)
    if not user or not valid:
        return jsonify({"success": False, "error": "Invalid username or password"}), 401
    return jsonify({"success": True, "token": _issue_token(user), "user": _public(user)})


@auth_bp.route("/me", methods=["GET"])
def me():
    claims = _read_token(_token_from_request())
    user = _find_user(claims["email"]) if claims else None
    if not user:
        return jsonify({"success": False, "error": "Please log in", "code": "auth_required"}), 401
    return jsonify({"success": True, "user": _public(user)})


@admin_bp.route("/users", methods=["GET"])
def list_users():
    """Accounts with their file count and storage use. Admin only. File contents are never exposed."""
    caller = _find_user(g.get("user_email", ""))
    if not caller or caller.get("role") != "admin":
        return jsonify({"success": False, "error": "Admins only"}), 403

    if cloud.enabled():
        rows = cloud._connect().execute(
            "SELECT u.id, u.username, u.email, u.name, u.role, u.created, "
            "COUNT(f.path) FILTER (WHERE NOT f.deleted AND right(f.path, 1) <> '/'), "
            "COALESCE(SUM(f.size) FILTER (WHERE NOT f.deleted), 0) "
            "FROM pc_users u LEFT JOIN pc_files f ON starts_with(f.path, 'users/' || u.id || '/') "
            "GROUP BY u.id ORDER BY u.id"
        ).fetchall()
        users = [
            {"id": r[0], "username": r[1], "email": r[2], "name": r[3], "role": r[4],
             "created": r[5].isoformat(), "file_count": int(r[6]), "bytes": int(r[7])}
            for r in rows
        ]
    else:
        with _file_lock:
            stored = _read_users().values()
        users = [{**_public(u), "created": None, "file_count": None, "bytes": None} for u in stored]
    return jsonify({"success": True, "users": users})
