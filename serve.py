#!/usr/bin/env python3
"""Serve a model with vLLM behind an API-key gateway, optionally via ngrok.

Layout:
    client --> [ngrok] --> gateway (server.host:server.port, checks API key)
                               --> vLLM (127.0.0.1:server.vllm_port)

Usage:
    export VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    python serve.py --config configs/default.yaml
    NGROK_AUTHTOKEN=... python serve.py --config configs/default.yaml --tunnel

The API key is read from the environment (never from the config file) and is
passed to vLLM through the environment, so it does not show up in `ps`.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import http.client
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "default.yaml"
MIN_KEY_LENGTH = 24

DEFAULTS: dict[str, Any] = {
    "model": "meta-models/Muse-Glimmer-30B",
    "vllm_args": {},
    "env": {},
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "vllm_port": 8001,
        "api_key_env": "VLLM_API_KEY",
        "public_paths": ["/health"],
        "allowed_prefixes": ["/v1", "/health", "/version", "/tokenize", "/detokenize"],
        "startup_timeout": 3600,
        "endpoint_file": "run/endpoint.json",
    },
    "tunnel": {
        "enabled": False,
        "authtoken_env": "NGROK_AUTHTOKEN",
        "domain": None,
    },
}

# Flags the launcher owns; setting them in vllm_args would bypass the gateway.
RESERVED_VLLM_ARGS = {"model", "host", "port", "api_key"}

log = logging.getLogger("serve")


class ServeError(Exception):
    """Configuration or startup failure with a user-facing message."""


# --------------------------------------------------------------------------- config


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ServeError(f"Config file not found: {path}")
    with path.open() as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ServeError(f"Config root must be a mapping: {path}")

    unknown = set(user_cfg) - set(DEFAULTS)
    if unknown:
        raise ServeError(f"Unknown top-level config keys: {sorted(unknown)}")

    cfg = _deep_merge(DEFAULTS, user_cfg)
    reserved = RESERVED_VLLM_ARGS & {k.replace("-", "_") for k in cfg["vllm_args"]}
    if reserved:
        raise ServeError(
            f"vllm_args must not set {sorted(reserved)}; use `model` and the "
            "`server` section instead."
        )
    if cfg["server"]["port"] == cfg["server"]["vllm_port"]:
        raise ServeError("server.port and server.vllm_port must differ.")
    return cfg


def to_cli_args(options: dict[str, Any]) -> list[str]:
    """Convert a vllm_args mapping into `vllm serve` flags.

    true -> --flag, false -> --no-flag, list -> --flag a b, dict -> --flag '<json>',
    null -> omitted.
    """
    args: list[str] = []
    for key, value in options.items():
        name = key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            args.append(f"--{name}" if value else f"--no-{name}")
        elif isinstance(value, dict):
            args += [f"--{name}", json.dumps(value)]
        elif isinstance(value, (list, tuple)):
            args += [f"--{name}", *map(str, value)]
        else:
            args += [f"--{name}", str(value)]
    return args


def vllm_executable() -> list[str]:
    # Jupyter kernels often have vllm importable without its script on PATH.
    exe = shutil.which("vllm")
    return [exe] if exe else [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def build_vllm_command(cfg: dict[str, Any]) -> list[str]:
    return [
        *vllm_executable(),
        "serve",
        cfg["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg["server"]["vllm_port"]),
        *to_cli_args(cfg["vllm_args"]),
    ]


def read_api_key(cfg: dict[str, Any]) -> str:
    env_name = cfg["server"]["api_key_env"]
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise ServeError(
            f"API key not set. Export {env_name} before starting, e.g.\n"
            f"  export {env_name}=\"$(python -c 'import secrets; "
            "print(secrets.token_urlsafe(32))')\""
        )
    if len(key) < MIN_KEY_LENGTH:
        message = f"{env_name} is shorter than {MIN_KEY_LENGTH} characters."
        if cfg["tunnel"]["enabled"]:
            raise ServeError(
                message + " Use a longer key before exposing the server publicly."
            )
        log.warning(message)
    return key


# --------------------------------------------------------------------------- processes


def start_vllm(cmd: list[str], cfg: dict[str, Any], api_key: str) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in cfg["env"].items()})
    env["VLLM_API_KEY"] = api_key
    log.info("Starting vLLM: %s", shlex.join(cmd))
    # New session so terminal Ctrl+C reaches only this launcher, which then
    # shuts vLLM (and its worker processes) down in order.
    return subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
        # Safe here: Popen runs before the gateway thread is started.
        preexec_fn=_exit_with_parent if sys.platform.startswith("linux") else None,  # noqa: PLW1509
    )


def _exit_with_parent() -> None:
    # If this launcher is killed outright (e.g. a Jupyter kernel restart),
    # SIGTERM vLLM too so it doesn't keep holding GPU memory.
    PR_SET_PDEATHSIG = 1
    try:
        ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except (OSError, AttributeError):
        pass  # best effort; normal shutdown still stops vLLM


def stop_vllm(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    log.info("Stopping vLLM (pid %d)...", proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("vLLM did not exit within %.0fs, killing it.", timeout)
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def wait_for_vllm(proc: subprocess.Popen, port: int, timeout: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    start = time.monotonic()
    next_notice = start + 60
    while True:
        if proc.poll() is not None:
            raise ServeError(f"vLLM exited during startup with code {proc.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    log.info("vLLM is healthy after %.0fs.", time.monotonic() - start)
                    return
        except (OSError, http.client.HTTPException):
            pass
        now = time.monotonic()
        if now - start > timeout:
            raise ServeError(
                f"vLLM not healthy after {timeout:.0f}s (server.startup_timeout)."
            )
        if now >= next_notice:
            log.info(
                "Waiting for vLLM to load the model (%.0fs elapsed)...", now - start
            )
            next_notice = now + 60
        time.sleep(2)


def start_gateway(cfg: dict[str, Any], api_key: str):
    import uvicorn

    from gateway import create_app

    server_cfg = cfg["server"]
    app = create_app(
        upstream_url=f"http://127.0.0.1:{server_cfg['vllm_port']}",
        api_key=api_key,
        public_paths=server_cfg["public_paths"],
        allowed_prefixes=server_cfg["allowed_prefixes"],
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=server_cfg["host"],
            port=server_cfg["port"],
            log_level="info",
            timeout_graceful_shutdown=5,
        )
    )
    # Off the main thread, uvicorn skips installing signal handlers, so
    # shutdown stays under this launcher's control.
    thread = threading.Thread(target=server.run, name="gateway", daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise ServeError(
                f"Gateway failed to start on {server_cfg['host']}:{server_cfg['port']} "
                "(is the port already in use?)."
            )
        time.sleep(0.1)
    return server, thread


def open_tunnel(cfg: dict[str, Any]) -> str:
    try:
        from pyngrok import conf, ngrok
        from pyngrok.exception import PyngrokError
    except ImportError as exc:
        raise ServeError(
            "Tunnel requested but pyngrok is not installed: pip install pyngrok"
        ) from exc

    tunnel_cfg = cfg["tunnel"]
    token = os.environ.get(tunnel_cfg["authtoken_env"])
    if token:
        conf.get_default().auth_token = token
    else:
        log.warning(
            "%s is not set; relying on an authtoken in the ngrok config file.",
            tunnel_cfg["authtoken_env"],
        )

    options = {"domain": tunnel_cfg["domain"]} if tunnel_cfg["domain"] else {}
    try:
        tunnel = ngrok.connect(f"127.0.0.1:{cfg['server']['port']}", "http", **options)
    except PyngrokError as exc:
        raise ServeError(f"Could not open ngrok tunnel: {exc}") from exc
    return tunnel.public_url


def close_tunnel() -> None:
    try:
        from pyngrok import ngrok

        ngrok.kill()
    except Exception as exc:  # noqa: BLE001 - best effort during shutdown
        log.warning("Error while stopping ngrok: %s", exc)


def write_endpoint_file(path: Path, info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n")


# --------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config (default: %(default)s)",
    )
    parser.add_argument("--model", help="Override `model` from the config")
    parser.add_argument("--port", type=int, help="Override server.port (gateway port)")
    parser.add_argument(
        "--tunnel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Open (or skip) the ngrok tunnel, overriding tunnel.enabled",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the vLLM command and exit"
    )
    return parser.parse_args(argv)


def _raise_on_sigterm(signum, _frame):
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [serve] %(levelname)s %(message)s"
    )
    logging.getLogger("pyngrok").setLevel(logging.WARNING)
    args = parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ServeError as exc:
        log.error("%s", exc)
        return 2
    if args.model:
        cfg["model"] = args.model
        # The config's served name belongs to the config's model.
        cfg["vllm_args"]["served_model_name"] = Path(
            args.model.rstrip("/")
        ).name.lower()
    if args.port is not None:
        cfg["server"]["port"] = args.port
    if args.tunnel is not None:
        cfg["tunnel"]["enabled"] = args.tunnel

    cmd = build_vllm_command(cfg)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    try:
        api_key = read_api_key(cfg)
    except ServeError as exc:
        log.error("%s", exc)
        return 2

    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    endpoint_file = cfg["server"]["endpoint_file"]
    endpoint_path = (SCRIPT_DIR / endpoint_file) if endpoint_file else None
    proc = gateway = gateway_thread = None
    tunnel_open = False
    exit_code = 0

    try:
        proc = start_vllm(cmd, cfg, api_key)
        wait_for_vllm(
            proc, cfg["server"]["vllm_port"], cfg["server"]["startup_timeout"]
        )
        gateway, gateway_thread = start_gateway(cfg, api_key)

        port = cfg["server"]["port"]
        served_name = cfg["vllm_args"].get("served_model_name") or cfg["model"]
        if isinstance(served_name, list):
            served_name = served_name[0]
        info = {
            "model": served_name,
            "local_url": f"http://127.0.0.1:{port}/v1",
            "public_url": None,
            "pid": os.getpid(),
        }
        if cfg["tunnel"]["enabled"]:
            tunnel_open = True
            info["public_url"] = open_tunnel(cfg) + "/v1"
        if endpoint_path:
            write_endpoint_file(endpoint_path, info)

        log.info("Ready. Model: %s", info["model"])
        log.info("  Local : %s", info["local_url"])
        if info["public_url"]:
            log.info("  Public: %s", info["public_url"])
        log.info("  Auth  : Authorization: Bearer $%s", cfg["server"]["api_key_env"])
        log.info("Press Ctrl+C (or send SIGTERM) to stop.")

        while True:
            if proc.poll() is not None:
                log.error("vLLM exited unexpectedly with code %s.", proc.returncode)
                exit_code = 1
                break
            if not gateway_thread.is_alive():
                log.error("Gateway stopped unexpectedly.")
                exit_code = 1
                break
            time.sleep(1)
    except ServeError as exc:
        log.error("%s", exc)
        exit_code = 1
    except (KeyboardInterrupt, SystemExit) as exc:
        log.info("Shutting down...")
        exit_code = (
            exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 0
        )
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if tunnel_open:
            close_tunnel()
        if gateway is not None:
            gateway.should_exit = True
            gateway_thread.join(timeout=10)
        if proc is not None:
            stop_vllm(proc)
        if endpoint_path:
            endpoint_path.unlink(missing_ok=True)
        log.info("Stopped.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
