"""HTTP Basic Auth. Required for the live server; tests may disable it."""
from __future__ import annotations

import secrets
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, username: str, password: str) -> None:
        super().__init__(app)
        self._user = username
        self._password = password

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        given = request.headers.get("authorization") or ""
        if given.lower().startswith("basic "):
            import base64
            try:
                decoded = base64.b64decode(given.split(" ", 1)[1]).decode("utf-8")
                user, _, password = decoded.partition(":")
            except Exception:                                      # noqa: BLE001
                user, password = "", ""
            if (secrets.compare_digest(user, self._user)
                    and secrets.compare_digest(password, self._password)):
                return await call_next(request)
        return Response(
            "Authentication required", status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="react-review"'})
