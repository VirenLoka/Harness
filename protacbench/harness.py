"""Driving the DeepSeek harness headlessly, one repair task per process.

The harness has no library entry point, so this is a subprocess driver. The
shape of it is fixed by what `dsh` actually offers, which is narrower than it
looks from the outside:

**Headless is a profile, not a flag.** `dsh --profile headless` is the
non-interactive mode. Its entire option set is `--json`, `--session-id` and
`--help`; there is no `--print`, no `--output-format`, no `--max-turns`.

**Launcher flags come first.** `--profile` and `--patch` are parsed by the
launcher, and the first token it does not recognise ends launcher parsing and
begins the profile's own argv. So `--json` goes after them, never before.

**The prompt goes in on stdin.** A positional task is joined on single spaces,
which quietly reformats a multi-line prompt; stdin is passed through verbatim.
Ours are multi-line, so stdin it is.

**There is no run timeout anywhere in the harness.** Nothing will stop a wedged
run, so the deadline here is ours and it is not optional.

**The model is not selectable from the command line.** Provider and model name
come from a YAML row, which is why this module writes a patch file rather than
assembling flags. Writing it from our own config is also what makes the whole
thing portable: nothing points at anyone's home directory, and the GPU box
gets the same file with its own endpoint in it.

The API key is never written into the patch. The patch carries `apiKeyEnv`,
the *name* of an environment variable, and the value is passed through the
subprocess environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from moltrain import logs

from .config import resolve

LOG = logs.get_logger("protacbench.harness")

# Events the headless runner emits on stdout under `--json`. The terminal
# `final` event is the only one exempt from the harness's 8 KiB string cap, so
# it is the one to trust for the answer itself.
EVENT_FINAL = "final"
EVENT_ERROR = "error"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_SESSION = "session"
EVENT_STATUS = "status"


class HarnessError(RuntimeError):
    """A problem running the harness, as opposed to a bad repair."""


@dataclass
class HarnessRun:
    """One headless invocation: what it answered and what it did to get there."""

    ok: bool
    final_text: str = ""
    session_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_failures: int = 0
    error: str | None = None
    exit_code: int | None = None
    wall_time_s: float = 0.0
    stderr_tail: str = ""

    @property
    def tool_names(self) -> list[str]:
        return [c.get("tool", "") for c in self.tool_calls]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "session_id": self.session_id,
            "tool_calls": len(self.tool_calls),
            "tool_failures": self.tool_failures,
            "tools_used": ";".join(sorted(set(self.tool_names))),
            "error": self.error,
            "exit_code": self.exit_code,
            "wall_time_s": round(self.wall_time_s, 2),
        }


# ------------------------------------------------------------------- the patch


def molbench_python(cfg: dict[str, Any]) -> str:
    """The interpreter that can import RDKit and FastMCP.

    Defaults to molbench's own virtualenv when it is there, because that is
    the one that definitely has them, and falls back to whatever is on PATH.
    """
    configured = cfg["harness"]["molbench_python"]
    if configured:
        return str(resolve(cfg, configured))
    venv = resolve(cfg, cfg["harness"]["molbench_path"]) / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else "python3"


def build_patch(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The cordis overlay: which model to use, and molbench as an MCP server.

    Returned as data so it can be asserted on in a test without writing a
    file or booting anything.
    """
    harness = cfg["harness"]
    model = harness["model"]
    molbench_dir = resolve(cfg, harness["molbench_path"])
    server = molbench_dir / "server.py"

    provider = str(model["provider"])

    patch: list[dict[str, Any]] = [
        # The OpenAI-compatible route. `apiKeyEnv` is the name of the variable
        # holding the key, never the key: this file is written into a run
        # directory and should stay safe to read.
        {
            "id": "llm-pi-ai",
            "config": {
                "providers": {
                    provider: {
                        "apiKeyEnv": model["api_key_env"],
                        "api": model["api"],
                        "baseURL": model["base_url"],
                        "models": [
                            {
                                "id": model["name"],
                                # Declared, because a model that reports no
                                # capacity inherits a fallback window twice
                                # the real size and compaction then fires too
                                # late to help.
                                "contextWindow": int(model["context_window"]),
                                "maxTokens": int(model["max_tokens"]),
                            }
                        ],
                    }
                }
            },
        },
        {
            "id": "agent-default-model",
            "config": {"provider": provider, "model": model["name"]},
        },
        {
            "insert": [
                {
                    "id": "mcp-molbench",
                    "name": "@deepseek-ai/dsh-mcp-client",
                    "config": {
                        "serverName": "molbench",
                        "transport": "stdio",
                        "command": molbench_python(cfg),
                        "args": [str(server)],
                        "cwd": str(molbench_dir),
                        # A dead MCP server is silently absent by default,
                        # which would show up as an agent that "chose" not to
                        # measure anything. Fail loudly instead.
                        "failOnStartupError": True,
                        "toolCallTimeoutMs": int(
                            float(harness["tool_timeout_s"]) * 1000
                        ),
                    },
                }
            ]
        },
    ]
    return patch


def write_patch(cfg: dict[str, Any]) -> Path:
    """Write the overlay and return its absolute path."""
    path = resolve(cfg, cfg["harness"]["patch_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(build_patch(cfg), sort_keys=False))
    LOG.debug("wrote harness patch to %s", path)
    return path


def build_command(cfg: dict[str, Any], patch_path: Path) -> list[str]:
    """The argv, with launcher flags ahead of profile flags.

    `--profile` and `--patch` belong to the launcher, and the first token it
    does not recognise ends its parsing, so `--json` has to come last.
    """
    harness = cfg["harness"]
    command = [harness["node_bin"]]
    if harness["node_bin"].endswith("npx"):
        command.append("-y")
    command.append(harness["dsh_package"])
    command += ["--profile", harness["profile"], "--patch", str(patch_path), "--json"]
    return command


def subprocess_env(cfg: dict[str, Any]) -> dict[str, str]:
    """The child environment: the API key, the permission mode, no telemetry."""
    harness = cfg["harness"]
    env = dict(os.environ)

    # `DSH_PERMISSION_MODE` drives two rows at once in the harness's base
    # config -- the sandbox mode and the approval policy -- so setting it is
    # how you keep them consistent. The default stays `workspace-write`.
    # `danger-full-access` is the only mode under which nothing can fail
    # closed, and it also switches the file sandbox off entirely, so it is
    # opt-in rather than the default.
    env["DSH_PERMISSION_MODE"] = str(harness["permission_mode"])
    env.setdefault("DSH_TELEMETRY_DISABLED", "1")

    if harness["dsh_home"]:
        env["DSH_HOME"] = str(resolve(cfg, harness["dsh_home"]))

    key_env = str(harness["model"]["api_key_env"])
    if key_env not in env and harness["require_api_key"]:
        raise HarnessError(
            f"{key_env} is not set, and harness.model.api_key_env names it as "
            "the variable holding the key for "
            f"{harness['model']['base_url']}. Export it, or point "
            "harness.model.api_key_env at the variable you do use."
        )
    return env


# ------------------------------------------------------------------ the parsing


def parse_events(stdout: str) -> list[dict[str, Any]]:
    """NDJSON, skipping anything that is not a JSON object.

    A line that does not parse is dropped rather than raising: the harness
    caps individual events at 32 KiB and marks them truncated, and one
    mangled status line should not lose the run's answer.
    """
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def summarise_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull out the answer, the tool calls, and any reported error."""
    final_text = ""
    session_id: str | None = None
    error: str | None = None
    tool_calls: list[dict[str, Any]] = []
    failures = 0

    for event in events:
        kind = event.get("type")
        if kind == EVENT_FINAL:
            final_text = str(event.get("text") or "")
        elif kind == EVENT_SESSION:
            session_id = event.get("sessionId")
        elif kind == EVENT_ERROR:
            error = str(event.get("message") or "unspecified harness error")
        elif kind == EVENT_TOOL_CALL:
            tool_calls.append(
                {
                    "callId": event.get("callId"),
                    "tool": event.get("tool"),
                    "input": event.get("input"),
                }
            )
        elif kind == EVENT_TOOL_RESULT:
            status = str(event.get("status") or "")
            if status and status != "completed":
                failures += 1

    return {
        "final_text": final_text,
        "session_id": session_id,
        "error": error,
        "tool_calls": tool_calls,
        "tool_failures": failures,
    }


# ------------------------------------------------------------------- the runner


async def run_prompt(
    cfg: dict[str, Any],
    prompt: str,
    *,
    patch_path: Path | None = None,
    timeout_s: float | None = None,
) -> HarnessRun:
    """One headless run. Never raises for a failed repair; raises only when
    the harness itself could not be started."""
    loop = asyncio.get_running_loop()
    started = loop.time()

    patch_path = patch_path or write_patch(cfg)
    command = build_command(cfg, patch_path)
    env = subprocess_env(cfg)
    deadline = float(timeout_s if timeout_s is not None else cfg["harness"]["timeout_s"])

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(resolve(cfg, cfg["harness"]["cwd"])),
        )
    except FileNotFoundError as exc:
        raise HarnessError(
            f"could not start {command[0]}. Install Node, or set "
            "harness.node_bin to an interpreter that can run the harness."
        ) from exc

    try:
        stdout_raw, stderr_raw = await asyncio.wait_for(
            process.communicate(prompt.encode()), timeout=deadline
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return HarnessRun(
            ok=False,
            error=f"the harness did not finish within {deadline}s",
            exit_code=None,
            wall_time_s=loop.time() - started,
        )

    stdout = stdout_raw.decode(errors="replace")
    stderr = stderr_raw.decode(errors="replace")
    summary = summarise_events(parse_events(stdout))

    # The harness exits 0 only when the turn ended `completed`. A turn that
    # failed mid-flight still emits a `final` event, often empty, and no
    # `error` event -- so the exit code is the signal, not the event stream.
    ok = process.returncode == 0 and not summary["error"]
    return HarnessRun(
        ok=ok,
        final_text=summary["final_text"],
        session_id=summary["session_id"],
        tool_calls=summary["tool_calls"],
        tool_failures=summary["tool_failures"],
        error=summary["error"]
        or (None if ok else f"harness exited {process.returncode}"),
        exit_code=process.returncode,
        wall_time_s=loop.time() - started,
        stderr_tail="\n".join(stderr.splitlines()[-20:]),
    )


# -------------------------------------------------------------------- the probe


@dataclass
class HarnessRuntime:
    """What this host can and cannot do, without running a model pass."""

    node_bin: str | None = None
    molbench_python: str | None = None
    molbench_server: str | None = None
    api_key_set: bool = False
    base_url: str = ""
    model: str = ""
    permission_mode: str = ""
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_bin": self.node_bin or "NOT FOUND",
            "molbench_python": self.molbench_python or "NOT FOUND",
            "molbench_server": self.molbench_server or "NOT FOUND",
            "api_key_set": self.api_key_set,
            "base_url": self.base_url,
            "model": self.model,
            "permission_mode": self.permission_mode,
        }


def probe(cfg: dict[str, Any]) -> HarnessRuntime:
    """Check everything a repair run needs, short of calling the model."""
    harness = cfg["harness"]
    model = harness["model"]
    runtime = HarnessRuntime(
        base_url=str(model["base_url"]),
        model=str(model["name"]),
        permission_mode=str(harness["permission_mode"]),
    )

    node = shutil.which(harness["node_bin"])
    runtime.node_bin = node
    if not node:
        runtime.problems.append(
            f"{harness['node_bin']} is not on PATH; the harness needs Node"
        )

    interpreter = molbench_python(cfg)
    if Path(interpreter).is_file() or shutil.which(interpreter):
        runtime.molbench_python = interpreter
    else:
        runtime.problems.append(f"molbench interpreter not found: {interpreter}")

    server = resolve(cfg, harness["molbench_path"]) / "server.py"
    if server.is_file():
        runtime.molbench_server = str(server)
    else:
        runtime.problems.append(f"molbench server not found: {server}")

    key_env = str(model["api_key_env"])
    runtime.api_key_set = bool(os.environ.get(key_env))
    if not runtime.api_key_set and harness["require_api_key"]:
        runtime.problems.append(f"{key_env} is not set")

    return runtime
