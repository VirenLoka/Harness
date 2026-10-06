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
import json
import logging
import re
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

log = logging.getLogger("gateway")

# Routes whose JSON body carries an output-token budget worth reshaping.
COMPLETION_PATHS = frozenset(
    {"/v1/chat/completions", "/v1/completions", "/v1/responses"}
)
# The field naming that budget differs per route; the first one present wins.
OUTPUT_TOKEN_FIELDS = ("max_completion_tokens", "max_tokens", "max_output_tokens")

# vLLM rejects a request whose prompt plus requested output exceeds the model's
# context. Both of its messages carry the numbers needed to retry at a size
# that fits. "at least" means the real prompt may be longer than reported.
CONTEXT_OVERFLOW = re.compile(
    r"maximum context length is (?P<total>\d+) tokens.*?"
    r"you requested (?P<output>\d+) output tokens.*?"
    r"prompt contains (?P<atleast>at least )?(?P<input>\d+) input tokens",
    re.DOTALL,
)
MIN_OUTPUT_TOKENS = 512
RETRY_SAFETY_MARGIN = 64


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


def _output_budget(payload: dict) -> tuple[str, int] | None:
    """The output-token field this request uses, and its value."""
    for field in OUTPUT_TOKEN_FIELDS:
        value = payload.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return field, value
    return None


def _fit_output_tokens(error_text: str, requested: int | None) -> int | None:
    """How many output tokens to ask for after a context-overflow refusal.

    With an exact prompt count the remaining room is known. With "at least"
    the prompt is only bounded below, so halve the ask instead and let a
    second attempt converge.
    """
    match = CONTEXT_OVERFLOW.search(error_text)
    if match is None:
        return None
    total = int(match.group("total"))
    reported_input = int(match.group("input"))
    asked = requested if requested is not None else int(match.group("output"))

    if match.group("atleast"):
        candidate = asked // 2
    else:
        candidate = total - reported_input - RETRY_SAFETY_MARGIN

    candidate = min(candidate, asked - 1)
    return candidate if candidate >= MIN_OUTPUT_TOKENS else None


def create_app(
    upstream_url: str,
    api_key: str,
    public_paths: Iterable[str] = ("/health",),
    allowed_prefixes: Iterable[str] = ("/v1",),
    max_output_tokens: int | None = None,
    context_retries: int = 3,
) -> Starlette:
    """Build the gateway ASGI app.

    Args:
        upstream_url: Base URL of the vLLM server, e.g. ``http://127.0.0.1:8001``.
        api_key: Key that clients must send as ``Authorization: Bearer <key>``.
        public_paths: Exact paths that may be called without a key.
        allowed_prefixes: Path prefixes forwarded to vLLM for authorized
            requests. Anything else returns 404. ``"/"`` allows everything.
        max_output_tokens: Optional ceiling applied to a completion request's
            output-token budget before it is forwarded. ``None`` forwards
            whatever the client asked for.
        context_retries: How many times to retry a request vLLM refused for
            exceeding the model's context, each time asking for fewer output
            tokens. ``0`` disables the retry.
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
        body_bytes = await request.body()
        url = httpx.URL(path=path, query=request.url.query.encode("ascii"))

        # Completion requests carry an output-token budget that can be reshaped
        # when it does not fit the model's context. Anything else is forwarded
        # byte for byte.
        payload: dict | None = None
        if request.method == "POST" and path in COMPLETION_PATHS and body_bytes:
            try:
                parsed = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            if isinstance(parsed, dict):
                payload = parsed

        if payload is not None and max_output_tokens is not None:
            budget = _output_budget(payload)
            if budget is not None and budget[1] > max_output_tokens:
                log.info(
                    "Capping %s from %d to %d for %s",
                    budget[0],
                    budget[1],
                    max_output_tokens,
                    path,
                )
                payload[budget[0]] = max_output_tokens
                body_bytes = json.dumps(payload).encode()

        attempts_left = context_retries if payload is not None else 0
        while True:
            try:
                upstream = await client.send(
                    client.build_request(
                        request.method, url, headers=headers, content=body_bytes
                    ),
                    stream=True,
                )
            except httpx.RequestError as exc:
                return _error(502, f"Model server unavailable: {exc}", "api_error")

            if upstream.status_code != 400 or attempts_left <= 0:
                break

            # A refusal body is small, so reading it costs nothing and may
            # tell us exactly how much output room the prompt left.
            error_text = (await upstream.aread()).decode(errors="replace")
            await upstream.aclose()
            budget = _output_budget(payload) if payload is not None else None
            fitted = _fit_output_tokens(error_text, budget[1] if budget else None)
            if fitted is None:
                return Response(
                    content=error_text,
                    status_code=400,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            field = budget[0] if budget else "max_tokens"
            log.warning(
                "vLLM refused %s for context overflow; retrying with %s=%d",
                path,
                field,
                fitted,
            )
            payload[field] = fitted  # type: ignore[index]
            body_bytes = json.dumps(payload).encode()
            attempts_left -= 1

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
