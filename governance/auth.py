import hmac
import time

import jwt

from core.config import settings

ROLES = {"admin": ("admin", lambda: settings.ADMIN_PASSWORD),
         "analyst": ("analyst", lambda: settings.ANALYST_PASSWORD)}


def authenticate(username: str, password: str) -> str | None:
    if username not in ROLES:
        return None
    role, pw = ROLES[username]
    return role if hmac.compare_digest(password, pw()) else None


def create_token(username: str, role: str, ttl: int = 3600) -> str:
    return jwt.encode({"sub": username, "role": role, "exp": int(time.time()) + ttl},
                      settings.JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
