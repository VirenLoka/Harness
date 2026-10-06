"""Gateway tests: auth, routing, and context-overflow recovery.

A fake upstream reproduces vLLM's real refusal: it rejects any request whose
prompt plus requested output exceeds the model's context, using vLLM 0.29's
exact message text.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path
from typing import Self

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway import _fit_output_tokens, _output_budget, create_app

API_KEY = "test-key-0123456789abcdefghijkl"
MAX_TOTAL = 131072
# The fake model's prompt length in tokens, chosen to mirror the real failure.
PROMPT_TOKENS = 98305


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_upstream(exact_count: bool, prompt_tokens: int = PROMPT_TOKENS) -> Starlette:
    """A stand-in vLLM that enforces prompt + max_tokens <= context."""
    seen: list[int] = []

    async def chat(request: Request) -> JSONResponse:
        payload = await request.json()
        requested = (
            payload.get("max_tokens") or payload.get("max_completion_tokens") or 16
        )
        seen.append(requested)
        if prompt_tokens + requested > MAX_TOTAL:
            qualifier = "" if exact_count else "at least "
            total = prompt_tokens + requested
            return JSONResponse(
                {
                    "message": (
                        f"This model's maximum context length is {MAX_TOTAL} tokens. "
                        f"However, you requested {requested} output tokens and your "
                        f"prompt contains {qualifier}{prompt_tokens} input tokens, for "
                        f"a total of {qualifier}{total} tokens. Please reduce the length "
                        "of the input prompt or the number of requested output tokens."
                    ),
                    "type": "BadRequestError",
                    "param": "input_tokens",
                    "code": 400,
                },
                status_code=400,
            )
        return JSONResponse(
            {"choices": [{"message": {"content": "ok"}}], "granted": requested}
        )

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({})

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/health", health),
        ]
    )
    app.state.seen = seen
    return app


class RunningUpstream:
    def __init__(self, app: Starlette):
        self.app = app
        self.port = free_port()
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="warning"
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> Self:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self.server.started:
            raise RuntimeError("fake upstream did not start")
        return self

    def __exit__(self, *_: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def post(
    app, body: dict, *, key: str | None = API_KEY, path: str = "/v1/chat/completions"
):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    # TestClient runs the lifespan, which is where the gateway opens its
    # upstream HTTP client.
    with TestClient(app) as client:
        return client.post(path, json=body, headers=headers)


# ------------------------------------------------------------------ unit tests


def test_output_budget_prefers_the_field_the_request_uses():
    assert _output_budget({"max_tokens": 100}) == ("max_tokens", 100)
    assert _output_budget({"max_completion_tokens": 50, "max_tokens": 100}) == (
        "max_completion_tokens",
        50,
    )
    assert _output_budget({"max_output_tokens": 7}) == ("max_output_tokens", 7)
    assert _output_budget({"temperature": 0.5}) is None
    # A bool is an int in Python; it is not a token budget.
    assert _output_budget({"max_tokens": True}) is None


def test_fit_uses_the_exact_remaining_room_when_the_prompt_is_known():
    message = (
        "This model's maximum context length is 131072 tokens. However, you "
        "requested 32768 output tokens and your prompt contains 120000 input "
        "tokens, for a total of 152768 tokens."
    )
    assert _fit_output_tokens(message, 32768) == 131072 - 120000 - 64


def test_fit_halves_the_ask_when_the_prompt_is_only_a_lower_bound():
    message = (
        "This model's maximum context length is 131072 tokens. However, you "
        "requested 32768 output tokens and your prompt contains at least 98305 "
        "input tokens, for a total of at least 131073 tokens."
    )
    assert _fit_output_tokens(message, 32768) == 16384


def test_fit_declines_when_no_useful_room_remains():
    message = (
        "This model's maximum context length is 131072 tokens. However, you "
        "requested 32768 output tokens and your prompt contains 131000 input "
        "tokens, for a total of 163768 tokens."
    )
    assert _fit_output_tokens(message, 32768) is None


def test_fit_ignores_unrelated_errors():
    assert _fit_output_tokens("some other failure", 100) is None


# ----------------------------------------------------------- integration tests


def test_retry_recovers_the_real_harness_failure():
    """32768 output against a 98305-token prompt: the exact reported error."""
    upstream = make_upstream(exact_count=False)
    with RunningUpstream(upstream) as running:
        app = create_app(f"http://127.0.0.1:{running.port}", API_KEY)
        response = post(app, {"model": "m", "messages": [], "max_tokens": 32768})

    assert response.status_code == 200, response.text
    # 32768 is refused; halving to 16384 fits inside the remaining room.
    assert upstream.state.seen == [32768, 16384]
    assert response.json()["granted"] == 16384


def test_retry_converges_on_a_much_longer_prompt():
    """A 125k-token prompt needs several halvings before the ask fits."""
    upstream = make_upstream(exact_count=False, prompt_tokens=125000)
    with RunningUpstream(upstream) as running:
        app = create_app(f"http://127.0.0.1:{running.port}", API_KEY)
        response = post(app, {"model": "m", "messages": [], "max_tokens": 32768})

    assert response.status_code == 200, response.text
    assert upstream.state.seen == [32768, 16384, 8192, 4096]
    assert response.json()["granted"] == 4096


def test_the_refusal_is_returned_once_the_retry_budget_is_spent():
    """A prompt that leaves almost no room still fails, with vLLM's own message."""
    upstream = make_upstream(exact_count=False, prompt_tokens=130500)
    with RunningUpstream(upstream) as running:
        app = create_app(f"http://127.0.0.1:{running.port}", API_KEY)
        response = post(app, {"model": "m", "messages": [], "max_tokens": 32768})

    assert response.status_code == 400
    assert upstream.state.seen == [32768, 16384, 8192, 4096]  # 1 attempt + 3 retries
    assert "maximum context length" in response.text


def test_retry_fits_exactly_when_the_prompt_length_is_known():
    upstream = make_upstream(exact_count=True)
    with RunningUpstream(upstream) as running:
        app = create_app(f"http://127.0.0.1:{running.port}", API_KEY)
        response = post(app, {"model": "m", "messages": [], "max_tokens": 32768})

    assert response.status_code == 200
    assert upstream.state.seen == [32768, MAX_TOTAL - PROMPT_TOKENS - 64]


def test_ceiling_prevents_the_refusal_without_any_retry():
    upstream = make_upstream(exact_count=False)
    with RunningUpstream(upstream) as running:
        app = create_app(
            f"http://127.0.0.1:{running.port}", API_KEY, max_output_tokens=8192
        )
        response = post(app, {"model": "m", "messages": [], "max_tokens": 32768})

    assert response.status_code == 200
    assert upstream.state.seen == [8192], (
        "the ceiling should make the first attempt fit"
    )


def test_retry_can_be_disabled():
    upstream = make_upstream(exact_count=False)
    with RunningUpstream(upstream) as running:
        app = create_app(f"http://127.0.0.1:{running.port}", API_KEY, context_retries=0)
        response = post(app, {"model": "m", "messages": [], "max_tokens": 32768})

    assert response.status_code == 400
    assert upstream.state.seen == [32768]
    assert "maximum context length" in response.text


def test_a_request_that_already_fits_is_forwarded_untouched():
    upstream = make_upstream(exact_count=False)
    with RunningUpstream(upstream) as running:
        app = create_app(f"http://127.0.0.1:{running.port}", API_KEY)
        response = post(app, {"model": "m", "messages": [], "max_tokens": 4096})

    assert response.status_code == 200
    assert upstream.state.seen == [4096]


def test_unauthorized_requests_never_reach_the_model():
    upstream = make_upstream(exact_count=False)
    with RunningUpstream(upstream) as running:
        app = create_app(f"http://127.0.0.1:{running.port}", API_KEY)
        response = post(app, {"model": "m", "max_tokens": 32768}, key="wrong")

    assert response.status_code == 401
    assert upstream.state.seen == []


def test_non_completion_routes_are_not_rewritten():
    upstream = make_upstream(exact_count=False)
    with RunningUpstream(upstream) as running:
        app = create_app(
            f"http://127.0.0.1:{running.port}", API_KEY, max_output_tokens=8192
        )

        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
