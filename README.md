# vLLM model server

Serves a Hugging Face model with vLLM through an OpenAI-compatible API. Every
request (except `GET /health`) needs an API key, and an ngrok tunnel can
optionally expose the server publicly. The default model is Meta's
[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)
(an agentic model that accepts text and images and supports tool calling),
sized for a single H200 and using its DFlash drafter for speculative decoding.

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
| `configs/default.yaml` | Muse Glimmer 30B on one H200: 131k context, reasoning and tool parsers, DFlash drafter |

## Setup

```bash
pip install -r requirements.txt
```

Muse Glimmer needs vLLM 0.29.0 or newer. If your container already ships a
recent enough vLLM built for its CUDA/torch version, only install `pyyaml` and
`pyngrok` (`python -c 'import vllm; print(vllm.__version__)'`). The gateway uses FastAPI/Starlette, uvicorn
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
python serve.py --model RedHatAI/Muse-Glimmer-30B-FP8-block   # served as muse-glimmer-30b-fp8-block
python serve.py --port 9000
python serve.py --dry-run                        # print the vllm command only
```

`--model` only swaps the checkpoint. The parsers and the drafter in the config
stay, so use it for other Muse Glimmer variants (the FP8 one above is untested
here) and write a separate config for other model families.

Once vLLM is healthy, the server prints the URLs and writes them to
`run/endpoint.json` (removed again on shutdown). Stop it with Ctrl+C or
`SIGTERM`; vLLM and the tunnel are shut down too. On Linux, vLLM also exits if
the launcher is killed outright, so a Jupyter kernel restart doesn't leave the
GPU occupied.

The first start downloads about 65 GB of weights (model plus drafter) into the
Hugging Face cache.
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
  -d '{"model": "muse-glimmer-30b",
       "messages": [{"role": "user", "content": "What is 17 * 23?"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-ngrok-domain>/v1", api_key="<your key>")
resp = client.chat.completions.create(
    model="muse-glimmer-30b",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
    max_tokens=8192,
    # low | medium | high | xhigh (the config's default is high)
    extra_body={"chat_template_kwargs": {"reasoning_strength": "low"}},
)
msg = resp.choices[0].message
print(getattr(msg, "reasoning", None))  # the model's reasoning (reasoning_parser)
print(msg.content)                      # the final answer
```

Tool calling uses the standard OpenAI `tools` parameter. The server turns the
model's own tool-call format into `tool_calls`:

```python
resp = client.chat.completions.create(
    model="muse-glimmer-30b",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
print(resp.choices[0].message.tool_calls)
```

Images go in as standard `image_url` content parts (an `https://` URL or a
`data:image/...;base64,` URI). The server downloads URL images itself. To limit
which hosts it will fetch from, set `allowed_media_domains` in the config.

Streaming (`stream=True`) goes through the gateway unbuffered. If a client
disconnects, the gateway closes the upstream request so vLLM stops generating.

The config sets Meta's recommended sampling (temperature 1.0, top_p 0.95,
top_k 64), and requests can override it. Set reasoning strength with
`chat_template_kwargs` as shown above, not by writing it into your system
prompt. The chat template already appends a `Reasoning strength:` line to the
system message.

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

## Quantization

Every key under `vllm_args` becomes a `vllm serve` flag, so vLLM's own
quantization options (`quantization`, `quantization_config`,
`kv_cache_dtype`) are set straight from the YAML. There are three ways to use
them:

- **FP8 at load time:** uncomment `quantization: fp8_per_channel` in
  `configs/default.yaml`. vLLM converts the BF16 text decoder to FP8 while
  loading. No separate checkpoint is needed, and the vision encoder and the
  drafter stay BF16. Weights drop from about 56 GiB to about 32 GiB.
  `fp8_per_block` and `fp8_per_tensor` also work (the config comments explain
  the difference).
- **A pre-quantized checkpoint:** point `model` (or `--model`) at it, for
  example `RedHatAI/Muse-Glimmer-30B-FP8-block` (untested here), and leave
  `quantization` unset. vLLM reads the scheme from the checkpoint, and setting
  both makes vLLM report a conflict.
- **FP8 KV cache:** `kv_cache_dtype: fp8`, together with
  `kv_cache_dtype_skip_layers: [sliding_window]` (see the config comments).

Some vLLM schemes don't fit this setup. `mxfp8` needs a Blackwell GPU to
quantize activations. On an H200 it falls back to weight-only FP8, and `mxfp4`
may keep activations in BF16.
`int8_per_channel_weight_only` and `nvfp4_per_token` only quantize
mixture-of-experts layers, which Muse Glimmer doesn't have, so they would
change nothing.

Run `python serve.py --dry-run` to confirm the flags, and check output quality
on your own prompts before switching over.

## Smaller GPUs

For a smaller GPU, copy the config, switch `model` to a quantized Muse Glimmer
checkpoint, and lower `max_model_len`. You can also delete `speculative_config`
to save the drafter's ~5 GB. Larger models can use `tensor_parallel_size`
across GPUs.
