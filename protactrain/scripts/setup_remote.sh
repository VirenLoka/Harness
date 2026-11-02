#!/usr/bin/env bash
#
# First-run setup for protactrain on the two-H200 box.
#
# What it does: creates the conda environment from environment.yml if it is
# not already there, then checks everything a run needs -- both GPUs, the
# sibling packages, the dataset, the libraries -- and prints a single report.
#
# What it does NOT do: run a model pass. Nothing here loads a checkpoint or
# touches the GPU beyond asking nvidia-smi what is present, so it is safe to
# run on a box that is already serving something.
#
# Usage:
#   protactrain/scripts/setup_remote.sh
#   protactrain/scripts/setup_remote.sh --no-create   # check only
#
# Safe to re-run. An existing environment is left exactly as it is -- this
# script never upgrades one, because a container that already ships a vLLM
# built against its own CUDA and torch is the normal case here and replacing
# that vLLM is how a working box stops working.
#
# It reports every problem rather than stopping at the first one. That is
# deliberate: the alternative is four round trips to a remote box to discover
# four missing things one at a time.

set -uo pipefail   # no -e: a failing check is the output, not the end of it

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_DIR="$ROOT/protactrain"
ENV_FILE="$PACKAGE_DIR/environment.yml"
ENV_NAME="protactrain"

CREATE=1
[ "${1:-}" = "--no-create" ] && CREATE=0

PROBLEMS=0

say()  { printf '\n==> %s\n' "$*"; }
ok()   { printf '  ok      %s\n' "$*"; }
note() { printf '          %s\n' "$*"; }
bad()  { printf '  MISSING %s\n' "$*"; PROBLEMS=$((PROBLEMS + 1)); }
warn() { printf '  !       %s\n' "$*"; }

# ------------------------------------------------------------- the environment

say "conda environment '$ENV_NAME'"

CONDA_BIN=""
for candidate in mamba conda micromamba; do
  if command -v "$candidate" >/dev/null 2>&1; then
    CONDA_BIN="$candidate"
    break
  fi
done

ENV_READY=0
if [ -z "$CONDA_BIN" ]; then
  bad "no conda, mamba or micromamba on PATH"
  note "install one, or activate the container's own python and re-run with --no-create"
elif "$CONDA_BIN" env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  ok "'$ENV_NAME' already exists; leaving it untouched"
  ENV_READY=1
elif [ "$CREATE" -eq 0 ]; then
  bad "'$ENV_NAME' does not exist (--no-create, so nothing was created)"
elif [ ! -f "$ENV_FILE" ]; then
  bad "$ENV_FILE"
else
  say "creating '$ENV_NAME' from $ENV_FILE (this takes a while)"
  if "$CONDA_BIN" env create -f "$ENV_FILE"; then
    ok "created '$ENV_NAME'"
    ENV_READY=1
  else
    bad "environment creation failed; the solver output above says why"
  fi
fi

# Every check below runs inside the environment when there is one, and in
# whatever python is on PATH when there is not -- a container with its own
# vLLM is a legitimate way to run this package, and the checks are still
# worth having there.
#
# The interpreter is resolved to a path rather than left as `conda run`,
# because each check is run under `timeout`, and timeout takes a command, not
# a shell function.
PYTHON_BIN="python"
if [ "$ENV_READY" -eq 1 ] && [ -n "$CONDA_BIN" ]; then
  RESOLVED="$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -1)"
  [ -x "$RESOLVED" ] && PYTHON_BIN="$RESOLVED"
fi

# moltrain's host checks learned this the hard way: peft's import chain
# reached scipy's BLAS loader, hung, and took the rest of the report down with
# it. A hanging import is one line here instead of the end of the run.
TIMEOUT=()
command -v timeout >/dev/null 2>&1 && TIMEOUT=(timeout 120)

run_py() {
  "${TIMEOUT[@]+"${TIMEOUT[@]}"}" "$PYTHON_BIN" "$@"
}

# --------------------------------------------------------------------- GPUs

say "GPUs"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi is not on PATH"
  note "protactrain refuses to run a model pass without a visible CUDA device,"
  note "so SFT and GRPO will not start on this host"
else
  GPU_LIST="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)"
  GPU_COUNT="$(printf '%s\n' "$GPU_LIST" | grep -c . )"
  if [ "$GPU_COUNT" -eq 0 ]; then
    bad "nvidia-smi reports no GPUs"
  else
    printf '%s\n' "$GPU_LIST" | while IFS= read -r line; do
      [ -n "$line" ] && ok "$line"
    done
    # grpo.ddp.world_size is 2 in configs/default.yaml. One card is a usable
    # smoke test (WORLD_SIZE=1 scripts/launch_ddp.sh grpo), so this is a
    # warning rather than a problem.
    if [ "$GPU_COUNT" -lt 2 ]; then
      warn "only $GPU_COUNT GPU visible; grpo.ddp.world_size is 2"
      note "run single-card with: WORLD_SIZE=1 protactrain/scripts/launch_ddp.sh grpo"
    fi
  fi
fi

# ------------------------------------------------------------- sibling packages
#
# protactrain imports moltrain (ATEM parsing, the completion mask, TRL
# compatibility, batched generation) and protacbench (every measurement), and
# neither is installed -- they are found on sys.path by sitting next to this
# package. So the check is "does the import work from the repo root", which is
# where launch_ddp.sh runs torchrun from.

say "sibling packages (from $ROOT)"

check_import() {
  local label="$1" code="$2"
  local out
  if out="$(cd "$ROOT" && run_py -c "$code" 2>&1)"; then
    ok "$label ${out:+-- $out}"
  else
    bad "$label"
    printf '%s\n' "$out" | tail -3 | sed 's/^/          /'
  fi
}

check_import "molbench tools" \
  'import protactrain, protactrain.toolkit as t
from protactrain.config import load_config
mod = t.import_molbench(load_config())
print(getattr(mod, "__name__", "molbench"))'

check_import "protacbench.grade" \
  'import protacbench.grade as g; print("SPAN_TARGET", g.SPAN_TARGET, "MAX_SA", g.MAX_SA)'

check_import "protacbench.score" \
  'import protacbench.score as s; print(len(s.WEIGHTS), "weighted terms")'

check_import "moltrain.atem" \
  'import moltrain.atem as a
print(a.parse_assistant_text("ok").__class__.__name__, "| stops", a.END_MESSAGE, a.END_TURN)'

check_import "moltrain.grpo_core" \
  'import moltrain.grpo_core as c; print(c.build_completion.__name__, "available")'

check_import "protactrain config" \
  'from protactrain.config import load_config
cfg = load_config()
print(cfg["model"]["name"], "| world_size", cfg["grpo"]["ddp"]["world_size"])'

# -------------------------------------------------------------------- libraries

say "libraries"

check_lib() {
  local label="$1" code="$2"
  local out
  if out="$(cd "$ROOT" && run_py -c "$code" 2>&1)"; then
    ok "$label $out"
  else
    bad "$label"
  fi
}

check_lib "rdkit       " 'import rdkit; print(rdkit.__version__)'
check_lib "torch       " 'import torch; print(torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())'
check_lib "transformers" 'import transformers; print(transformers.__version__)'
check_lib "trl         " 'import trl; print(trl.__version__)'
check_lib "peft        " 'import peft; print(peft.__version__)'
check_lib "accelerate  " 'import accelerate; print(accelerate.__version__)'
check_lib "vllm        " 'import vllm; print(vllm.__version__)'

# --------------------------------------------------------------------- dataset

say "dataset"

# dataset.repair_tasks in configs/default.yaml. 2,229 rows across 346 targets,
# all of stage 'generated'. Note that its `paths` column points at
# /home/jovyan/... on the box that produced it; nothing in this package
# requires those files to exist.
TASKS_FILE="$ROOT/repair_tasks.jsonl"
if [ -s "$TASKS_FILE" ]; then
  ROWS="$(wc -l < "$TASKS_FILE" | tr -d ' ')"
  ok "repair_tasks.jsonl ($ROWS rows)"
  if [ "$ROWS" -ne 2229 ]; then
    warn "expected 2229 rows; this file has $ROWS"
  fi
else
  bad "$TASKS_FILE"
  note "produced by: python -m protacbench assess"
fi

# ---------------------------------------------------------------------- report

say "summary"

if [ "$PROBLEMS" -eq 0 ]; then
  printf '  nothing missing.\n'
  cat <<'EOF'

  next, in order -- none of these runs a model pass except the last two:

    python -m protactrain.tasks probe
    python -m protactrain.tasks build
    python -m protactrain.sft check
    python -m protactrain.grpo check

    python -m protactrain.traces collect --source scripted
    protactrain/scripts/launch_ddp.sh sft
    protactrain/scripts/launch_ddp.sh grpo
EOF
else
  printf '  %s problem(s) above. Fix them and re-run; this script is idempotent.\n' "$PROBLEMS"
fi
printf '\n'

exit 0
