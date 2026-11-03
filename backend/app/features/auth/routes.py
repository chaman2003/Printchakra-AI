"""
Accounts: sign up, log in, and protect the API.

Passwords are hashed with werkzeug (scrypt). A login returns a signed token (7 days) that the
frontend sends as `Authorization: Bearer <token>`. Users live in Postgres when DATABASE_URL is set
(cloud) and in a JSON file otherwise (local development).

shortcut: all users share one workspace (files are not per-user), add a user_id to the stored paths
when real multi-tenancy is needed. There is also no login rate limit beyond the platform's own.
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

TOKEN_MAX_AGE_S = 7 * 24 * 3600
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 6

# Demo account so the deployed app can be tried straight away. Remove with SEED_TEST_USER=0.
SEED_EMAIL = "test@gmail.com"
SEED_PASSWORD = "test123"

_file_lock = threading.Lock()
_seeded = False


# ---------------------------------------------------------------- storage

def _users_file() -> str:
    from app.core.config import get_data_dirs

    return os.path.join(get_data_dirs()["DATA_DIR"], "users.json")


def _find_user(email: str):
    """Return {id, email, name, password_hash} or None."""
    if cloud.enabled():
        row = cloud._connect().execute(
            "SELECT id, email, name, password_hash FROM pc_users WHERE email = %s", (email,)
        ).fetchone()
        return dict(zip(("id", "email", "name", "password_hash"), row)) if row else None
    with _file_lock:
        try:
            with open(_users_file(), "r", encoding="utf-8") as handle:
                return json.load(handle).get(email)
        except (OSError, ValueError):
            return None


def _create_user(email: str, name: str, password: str):
    """Create a user. Returns the user dict, or None when the email is already taken."""
    password_hash = generate_password_hash(password)
    if cloud.enabled():
        row = cloud._connect().execute(
            "INSERT INTO pc_users (email, name, password_hash) VALUES (%s, %s, %s) "
            "ON CONFLICT (email) DO NOTHING RETURNING id",
            (email, name, password_hash),
        ).fetchone()
        return {"id": row[0], "email": email, "name": name} if row else None
    with _file_lock:
        path = _users_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                users = json.load(handle)
        except (OSError, ValueError):
            users = {}
        if email in users:
            return None
        users[email] = {"id": len(users) + 1, "email": email, "name": name, "password_hash": password_hash}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(users, handle)
        return {"id": users[email]["id"], "email": email, "name": name}


def _ensure_seed_user() -> None:
    global _seeded
    if _seeded or os.environ.get("SEED_TEST_USER", "1") == "0":
        return
    try:
        if _find_user(SEED_EMAIL) is None:
            _create_user(SEED_EMAIL, "Test User", SEED_PASSWORD)
        _seeded = True
    except Exception as exc:
        logger.error("Could not seed the test user: %s", exc)


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
    return {"id": user["id"], "email": user["email"], "name": user["name"]}


# ---------------------------------------------------------------- guard

# Paths reachable without logging in.
_OPEN_PREFIXES = ("/auth/", "/health", "/ping")


def require_login():
    """before_request hook: reject API calls that carry no valid token."""
    if request.method == "OPTIONS" or request.path == "/" or request.path.startswith(_OPEN_PREFIXES):
        return None
    claims = _read_token(_token_from_request())
    if not claims:
        return jsonify({"success": False, "error": "Please log in", "code": "auth_required"}), 401
    g.user_email = claims["email"]
    return None


# ---------------------------------------------------------------- routes

def _body() -> dict:
    return request.get_json(silent=True) or {}


@auth_bp.route("/signup", methods=["POST"])
def signup():
    _ensure_seed_user()
    data = _body()
    name = str(data.get("name") or "").strip()[:80]
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")

    if not name:
        return jsonify({"success": False, "error": "Please enter your name"}), 400
    if not EMAIL_RE.match(email) or len(email) > 254:
        return jsonify({"success": False, "error": "Please enter a valid email address"}), 400
    if len(password) < MIN_PASSWORD or len(password) > 200:
        return jsonify({"success": False, "error": f"Password must be at least {MIN_PASSWORD} characters"}), 400

    user = _create_user(email, name, password)
    if user is None:
        return jsonify({"success": False, "error": "An account with this email already exists"}), 409
    return jsonify({"success": True, "token": _issue_token(user), "user": _public(user)}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    _ensure_seed_user()
    data = _body()
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")

    user = _find_user(email) if email else None
    # Always run one hash check so a wrong email takes as long as a wrong password.
    valid = check_password_hash(user["password_hash"] if user else generate_password_hash("x"), password)
    if not user or not valid:
        return jsonify({"success": False, "error": "Invalid email or password"}), 401
    return jsonify({"success": True, "token": _issue_token(user), "user": _public(user)})


@auth_bp.route("/me", methods=["GET"])
def me():
    claims = _read_token(_token_from_request())
    user = _find_user(claims["email"]) if claims else None
    if not user:
        return jsonify({"success": False, "error": "Please log in", "code": "auth_required"}), 401
    return jsonify({"success": True, "user": _public(user)})
