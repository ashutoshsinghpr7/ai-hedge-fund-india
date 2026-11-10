"""Kite Connect authentication and access-token lifecycle.

Kite access tokens are invalidated every morning (~6 AM IST), so a session file
is kept between runs and refreshed by `hedgefund login`. The API secret is read
from the environment and is never written to disk.
"""

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

SESSION_FILE = Path(__file__).resolve().parent.parent.parent / ".kite_session.json"


class KiteAuthRequired(RuntimeError):
    """Raised when no usable Kite access token is available."""


def login_url(api_key: str) -> str:
    """URL the user must open to authorise the app for the day."""
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def extract_request_token(value: str) -> str:
    """Accept a bare request_token or the full redirect URL pasted from the browser."""
    value = value.strip()
    if "request_token" in value:
        query = urlparse(value).query or value
        tokens = parse_qs(query).get("request_token")
        if tokens:
            return tokens[0]
    return value


def load_session(path: Path | str = SESSION_FILE) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable Kite session file %s: %s", path, exc)
        return None


def save_session(session: dict, path: Path | str = SESSION_FILE) -> None:
    path = Path(path)
    # `default=str` keeps the write from failing on non-JSON values such as the
    # datetimes Kite returns in the session payload.
    path.write_text(json.dumps(session, indent=2, default=str), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # best effort on Windows
        pass


def _as_timestamp(value) -> str:
    """Normalise Kite's datetime/date login_time to a JSON-safe ISO string."""
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).isoformat(timespec="seconds")
    if value:
        return str(value)
    return datetime.now().isoformat(timespec="seconds")


def complete_login(
    request_token: str,
    api_key: str,
    api_secret: str,
    path: Path | str = SESSION_FILE,
) -> dict:
    """Exchange a request_token for an access_token and persist the session."""
    client = KiteConnect(api_key=api_key)
    data = client.generate_session(extract_request_token(request_token), api_secret=api_secret)
    session = {
        "api_key": api_key,
        "access_token": data["access_token"],
        "user_id": data.get("user_id"),
        "login_time": _as_timestamp(data.get("login_time")),
    }
    save_session(session, path)
    return session


def create_client(
    api_key: Optional[str] = None,
    access_token: Optional[str] = None,
    session_path: Path | str = SESSION_FILE,
    require_session: bool = True,
) -> KiteConnect:
    """Build an authenticated Kite client from args, env, or the session file."""
    api_key = api_key or os.environ.get("KITE_API_KEY")
    if not api_key:
        raise KiteAuthRequired("KITE_API_KEY is not set. Add it to .env (see .env.example).")

    access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN")
    if not access_token:
        session = load_session(session_path)
        if session and session.get("api_key") in (None, api_key):
            access_token = session.get("access_token")

    client = KiteConnect(api_key=api_key)
    if access_token:
        client.set_access_token(access_token)
    elif require_session:
        raise KiteAuthRequired(
            "No Kite access token found. Run `hedgefund login` (or set KITE_ACCESS_TOKEN). "
            f"Tokens expire every morning, so re-run login at the start of each trading day.\n{login_url(api_key)}"
        )
    return client
