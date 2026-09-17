"""API-key gateway that sits in front of the vLLM server.

vLLM's own ``--api-key`` only guards routes under /v1, /v2, /inference and
/cohere. Other routes, notably the SageMaker-compatible ``/invocations`` (which
runs chat completions), are served without a key. Exposing vLLM directly via
ngrok would therefore let anyone use the model. This gateway requires the key
on every request except an explicit list of public paths, and only forwards
paths under an allowlist of prefixes.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# Headers that describe a single hop and must not be forwarded.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

ALL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def _error(status: int, message: str, err_type: str, **headers: str) -> JSONResponse:
    # OpenAI-style error body so OpenAI SDK clients surface a readable message.
    return JSONResponse(
        {"error": {"message": message, "type": err_type, "code": status}},
        status_code=status,
        headers=headers or None,
    )


def _matches_prefix(path: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        prefix = prefix.rstrip("/")
        if not prefix or path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def create_app(
    upstream_url: str,
    api_key: str,
    public_paths: Iterable[str] = ("/health",),
    allowed_prefixes: Iterable[str] = ("/v1",),
) -> Starlette:
    """Build the gateway ASGI app.

    Args:
        upstream_url: Base URL of the vLLM server, e.g. ``http://127.0.0.1:8001``.
        api_key: Key that clients must send as ``Authorization: Bearer <key>``.
        public_paths: Exact paths that may be called without a key.
        allowed_prefixes: Path prefixes forwarded to vLLM for authorized
            requests. Anything else returns 404. ``"/"`` allows everything.
    """
    key_digest = hashlib.sha256(api_key.encode()).digest()
    public = frozenset(public_paths)
    allowed = tuple(allowed_prefixes)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        # Reasoning models can generate for many minutes, so no read timeout.
        app.state.client = httpx.AsyncClient(
            base_url=upstream_url,
            timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    def authorized(request: Request) -> bool:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer":
            return False
        token_digest = hashlib.sha256(token.strip().encode()).digest()
        return hmac.compare_digest(token_digest, key_digest)

    async def proxy(request: Request) -> Response:
        path = request.url.path
        # httpx normalizes dot segments, so "/health/../invocations" would reach
        # a different route upstream than the one checked here. Reject them.
        if any(segment in (".", "..") for segment in path.split("/")):
            return _error(400, "Invalid request path.", "invalid_request_error")

        if path not in public:
            if not authorized(request):
                return _error(
                    401,
                    "Invalid or missing API key. Send 'Authorization: Bearer <key>'.",
                    "authentication_error",
                    **{"WWW-Authenticate": "Bearer"},
                )
            if not _matches_prefix(path, allowed):
                return _error(404, f"Route {path} is not exposed.", "not_found_error")

        client: httpx.AsyncClient = request.app.state.client
        headers = [
            (name, value)
            for name, value in request.headers.raw
            if name.decode("latin-1").lower() not in HOP_BY_HOP_HEADERS
        ]
        upstream_request = client.build_request(
            request.method,
            httpx.URL(path=path, query=request.url.query.encode("ascii")),
            headers=headers,
            content=await request.body(),
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.RequestError as exc:
            return _error(502, f"Model server unavailable: {exc}", "api_error")

        async def body() -> AsyncIterator[bytes]:
            # Closing the upstream response when the client disconnects makes
            # vLLM abort the generation instead of running it to completion.
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in HOP_BY_HOP_HEADERS
        }
        return StreamingResponse(
            body(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream.aclose),
        )

    return Starlette(
        routes=[
            Route("/", proxy, methods=ALL_METHODS),
            Route("/{path:path}", proxy, methods=ALL_METHODS),
        ],
        lifespan=lifespan,
    )
