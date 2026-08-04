"""Request-size and rate limiting.

The server previously accepted a body of any size and an unlimited request
rate. Combined with the old defaults — bound to 0.0.0.0, CORS `*`, and an auth
middleware that was never mounted — a single unauthenticated caller could
exhaust memory or pin the process. Both limits are cheap and belong at the
edge, before any handler allocates.
"""

from __future__ import annotations

import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies using the declared Content-Length.

    This is the cheap pre-check: it costs nothing and rejects the honest case
    before a byte is buffered. Handlers that stream still enforce their own
    limit for callers that lie about (or omit) Content-Length.
    """

    def __init__(self, app: object, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: object):
        if self._max_bytes > 0:
            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self._max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "payload_too_large",
                        "detail": f"Body exceeds the {self._max_bytes} byte limit",
                    },
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window-free sliding request cap per client address.

    Deliberately in-process and approximate: the goal is to stop one runaway
    agent loop or a trivial denial-of-service, not to be a distributed quota
    system. Health endpoints are exempt so a limiter cannot make a service look
    down to its orchestrator.
    """

    _EXEMPT = frozenset({"/", "/health"})

    def __init__(self, app: object, per_minute: int) -> None:
        super().__init__(app)
        self._per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}

    async def dispatch(self, request: Request, call_next: object):
        if self._per_minute <= 0 or request.url.path in self._EXEMPT:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._hits.setdefault(client, deque())
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._per_minute:
            retry_after = max(1, int(60.0 - (now - window[0])))
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": "rate_limited",
                    "detail": f"More than {self._per_minute} requests per minute",
                },
            )

        window.append(now)
        # Bound the bookkeeping: drop clients that have gone quiet.
        if len(self._hits) > 10_000:
            self._hits = {
                k: v for k, v in self._hits.items() if v and v[-1] >= cutoff
            }
        return await call_next(request)
