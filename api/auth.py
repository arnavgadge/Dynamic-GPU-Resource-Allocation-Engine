"""A demonstration login mechanism - not enterprise authentication.

Five seeded accounts (one ADMIN, four USER), an opaque in-memory
session token per login, and one rule that matters for correctness
(Part 29 of the brief): **a request is never trusted to say "I am
User A" on its own** - every authenticated route resolves the caller
from their session token, looked up here, never from a client-
supplied user id in the request body.

No passwords, no OAuth, no database - this is a course-project
demonstration of "the backend knows who is asking", not a security
product.
"""

import secrets
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Account:
    username: str
    role: str  # "ADMIN" or "USER"
    display_name: str


ACCOUNTS: Dict[str, Account] = {
    "admin": Account(username="admin", role="ADMIN", display_name="Admin"),
    "user_a": Account(username="user_a", role="USER", display_name="User A"),
    "user_b": Account(username="user_b", role="USER", display_name="User B"),
    "user_c": Account(username="user_c", role="USER", display_name="User C"),
    "user_d": Account(username="user_d", role="USER", display_name="User D"),
}

#: token -> username. Cleared on process restart, exactly like the
#: rest of this project's in-memory state - there is no persistence
#: layer anywhere in this system, on purpose (Part 48).
_SESSIONS: Dict[str, str] = {}


def login(username: str) -> Account:
    """Start a session for a seeded demo account.

    Raises ``KeyError`` for an unrecognized username - the caller
    (the login route) maps that to a 401, not a fabricated account.
    """
    if username not in ACCOUNTS:
        raise KeyError(f"unknown demo account {username!r}")
    return ACCOUNTS[username]


def create_token(username: str) -> str:
    token = secrets.token_urlsafe(24)
    _SESSIONS[token] = username
    return token


def resolve_token(token: str) -> Optional[Account]:
    """The account a session token belongs to, or ``None`` if the
    token is missing/unrecognized/expired (this process restarted)."""
    username = _SESSIONS.get(token)
    if username is None:
        return None
    return ACCOUNTS.get(username)


def logout(token: str) -> None:
    _SESSIONS.pop(token, None)
