# vLLM model server

Serves a Hugging Face model with vLLM through an OpenAI-compatible API. Every
request (except `GET /health`) needs an API key, and an ngrok tunnel can
optionally expose the server publicly. The default model is
`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`, sized for a single H200.

```
client ──► [ngrok] ──► gateway :8000  (checks Bearer key on every route)
                           └──► vLLM 127.0.0.1:8001
```

The gateway (`gateway.py`) exists because vLLM's `--api-key` only covers
`/v1`, `/v2`, `/inference` and `/cohere`. Other routes are open, including
`/invocations`, which runs chat completions. Pointing ngrok straight at vLLM
would let anyone with the URL use the model. The gateway also forwards only
`allowed_prefixes`, so key holders can't reach admin or dev endpoints either.

## Files

| File | Purpose |
| --- | --- |
| `serve.py` | Launcher: reads the config, starts vLLM, waits for it, starts the gateway, opens the tunnel, cleans everything up on exit |
| `gateway.py` | API-key reverse proxy (Starlette + httpx, streaming) |
| `configs/default.yaml` | DeepSeek-R1-Distill-Qwen-32B on one H200 |

## Setup

```bash
pip install -r requirements.txt
```

If your container already ships vLLM built for its CUDA/torch version, only
install `pyyaml` and `pyngrok`. The gateway uses FastAPI/Starlette, uvicorn
and httpx, which vLLM already installs.

Generate an API key once and keep it somewhere safe:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

## Run

```bash
export VLLM_API_KEY=<your key>
python serve.py                                  # uses configs/default.yaml
```

Options:

```bash
python serve.py --config configs/other.yaml
python serve.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   # served as deepseek-r1-distill-qwen-14b
python serve.py --port 9000
python serve.py --dry-run                        # print the vllm command only
```

Once vLLM is healthy, the server prints the URLs and writes them to
`run/endpoint.json` (removed again on shutdown). Stop it with Ctrl+C or
`SIGTERM`; vLLM and the tunnel are shut down too. On Linux, vLLM also exits if
the launcher is killed outright, so a Jupyter kernel restart doesn't leave the
GPU occupied.

The first start downloads about 65 GB of weights into the Hugging Face cache.
Set `HF_HOME` (or `env.HF_HOME` in the config) to a volume that persists.

### With ngrok

```bash
export VLLM_API_KEY=<your key>
export NGROK_AUTHTOKEN=<token from dashboard.ngrok.com>
python serve.py --tunnel
```

pyngrok downloads the ngrok agent binary on first use. To keep the same URL
across restarts, reserve a domain in the ngrok dashboard and set
`tunnel.domain`. With `--tunnel`, the API key must be at least 24 characters.

### From a Jupyter notebook

Running `!python serve.py` blocks the cell, so start it in the background:

```python
import json, os, subprocess, sys, time

os.environ["VLLM_API_KEY"] = "..."        # or set it in the container env
os.environ["NGROK_AUTHTOKEN"] = "..."
os.makedirs("run", exist_ok=True)

server = subprocess.Popen(
    [sys.executable, "serve.py", "--tunnel"],
    stdout=open("run/serve.log", "w"), stderr=subprocess.STDOUT,
)

while not os.path.exists("run/endpoint.json"):   # model load takes a few minutes
    assert server.poll() is None, "server exited, see run/serve.log"
    time.sleep(5)
print(json.load(open("run/endpoint.json")))
```

```python
server.terminate(); server.wait()          # stop it
```

Follow progress with `!tail -n 20 run/serve.log`.

## Calling the API

```bash
curl https://<your-ngrok-domain>/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-r1-distill-qwen-32b",
       "messages": [{"role": "user", "content": "What is 17 * 23?"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-ngrok-domain>/v1", api_key="<your key>")
resp = client.chat.completions.create(
    model="deepseek-r1-distill-qwen-32b",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
    max_tokens=8192,
)
msg = resp.choices[0].message
print(getattr(msg, "reasoning_content", None))  # the <think> section (reasoning_parser)
print(msg.content)                               # the final answer
```

Streaming (`stream=True`) goes through the gateway unbuffered. If a client
disconnects, the gateway closes the upstream request so vLLM stops generating.

DeepSeek recommends putting all instructions in the user message for R1
distills rather than using a system prompt. The config sets their suggested
sampling defaults (temperature 0.6, top_p 0.95), and requests can override them.

## Config

See `configs/default.yaml` for every option with comments. The main sections:

- `model`: Hugging Face repo id or local path.
- `vllm_args`: passed to `vllm serve` (`snake_case` becomes `--kebab-case`;
  `false` becomes `--no-<flag>`). `host`, `port` and `api_key` are rejected
  here because the launcher controls them.
- `env`: extra environment variables for the vLLM process.
- `server`: gateway bind address and port, internal vLLM port, the name of the
  env var holding the key, public paths, allowed prefixes, and startup timeout.
- `tunnel`: `enabled`, the name of the ngrok authtoken env var, and an
  optional reserved `domain`.

The API key is never read from the config file. It comes from the environment
variable named in `server.api_key_env` and is handed to vLLM through its
environment, so it doesn't appear in `ps` output.

For a smaller GPU, copy the config and change `model`, `served_model_name` and
`max_model_len`. Larger models can use `tensor_parallel_size` across GPUs.
