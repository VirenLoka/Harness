#!/usr/bin/env bash
#
# Launch an SFT or GRPO run across the two H200s with torchrun.
#
# Usage:
#   scripts/launch_ddp.sh <sft|grpo> [config.yaml] [extra args...]
#   scripts/launch_ddp.sh grpo --dry-run
#
# Examples:
#   scripts/launch_ddp.sh sft
#   scripts/launch_ddp.sh grpo protactrain/configs/default.yaml --verbose
#   WORLD_SIZE=1 scripts/launch_ddp.sh grpo --dry-run
#
# Everything after the stage and the optional config path is handed to the
# stage's own CLI untouched, so --verbose, --quiet and --log-file work here
# exactly as they do when the module is run directly.
#
# This script runs a model pass. It is for the GPU host and nothing else;
# `python -m protactrain.grpo check` is the thing to run first.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_DIR="$ROOT/protactrain"
DEFAULT_CONFIG="$PACKAGE_DIR/configs/default.yaml"

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

# ------------------------------------------------------------------ arguments

[ "$#" -ge 1 ] || die "usage: scripts/launch_ddp.sh <sft|grpo> [config.yaml] [extra args...]"

STAGE="$1"; shift
case "$STAGE" in
  sft|grpo) ;;
  *) die "stage must be 'sft' or 'grpo', not '$STAGE'" ;;
esac

# The config is positional only when it does not start with a dash, so
# `launch_ddp.sh grpo --dry-run` is unambiguous.
CONFIG="$DEFAULT_CONFIG"
if [ "$#" -ge 1 ] && [ "${1#-}" = "$1" ]; then
  CONFIG="$1"; shift
fi
[ -f "$CONFIG" ] || die "config not found: $CONFIG"

DRY_RUN=0
EXTRA=()
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    # Consumed here rather than forwarded: this is the launcher's dry run,
    # which prints the torchrun command and never starts a process. The
    # stages have a --dry-run of their own that builds everything and skips
    # Trainer.train(); pass it as --stage-dry-run if that is what you want.
    DRY_RUN=1
  elif [ "$arg" = "--stage-dry-run" ]; then
    EXTRA+=("--dry-run")
  else
    EXTRA+=("$arg")
  fi
done

# -------------------------------------------------------------- thread caps
#
# A JupyterLab container exposes every core on the host while capping how many
# threads one process may create, and OpenBLAS, OMP and MKL each size their
# pool from the visible core count. The first BLAS-backed import then dies
# with `blas_thread_init: pthread_create failed for thread 63 of 64`.
# protactrain/__init__.py caps these on import, but that is too late for
# anything torchrun itself loads, and two ranks on one host double the
# demand -- so they are set here, before the launcher starts.
#
# An explicitly set variable always wins, matching the package's own default.
#
# This is a budget for the WHOLE job, divided across the ranks further down.
# It used to be applied per rank, so `--nproc_per_node=2` asked for twice what
# was intended and a two-rank SFT run died with
#
#   libgomp: Thread creation failed: Resource temporarily unavailable
#   error: can't start new thread
#   ... exitcode: -11 (SIGSEGV)
#
# EAGAIN from pthread_create, i.e. the container's task cap. The segfault is
# libgomp failing to recover, not a separate fault worth chasing.
THREAD_BUDGET="${PROTACTRAIN_NUM_THREADS:-8}"

# Tokenizers forks a thread pool per worker and warns loudly once the data
# collator forks; the fast tokenizer is not the bottleneck in a rollout-bound
# step, so the quiet setting is the right one.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# --------------------------------------------------------------- fork poisoning
#
# vLLM forks its EngineCore child by default, and a forked child cannot touch
# CUDA once the parent has a context -- it dies with "Cannot re-initialize CUDA
# in forked subprocess". `spawn` gives the child a clean interpreter, so
# nothing the parent imported first can break it. `PYTORCH_NVML_BASED_CUDA_CHECK`
# keeps `torch.cuda.is_available()` off the `cuInit` path that creates the
# context in the first place.
#
# protactrain/__init__.py sets both, but that is too late for anything torchrun
# loads before the package is imported, which is the same reason the thread
# caps are repeated here.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"

# ------------------------------------------------------------------- ranks
#
# The config decides the rank count; WORLD_SIZE overrides it for a one-off.
#
# This used to be a hardcoded `${WORLD_SIZE:-2}`, which made `grpo.ddp` in
# the config documentation rather than configuration: setting
# `world_size: 1` there changed nothing and the second rank started anyway
# -- on device 1, because `local_rank` is the device, which is exactly the
# card a single-card run is trying to stay off.
#
# Asked of the package rather than grepped out of the YAML, because a
# config that omits the key still has a value through DEFAULTS, and
# because `world_size if enabled else 1` is the rule `grpo check` already
# plans with.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON=python3; else PYTHON=python; fi
fi

if [ -n "${WORLD_SIZE:-}" ]; then
  NPROC="$WORLD_SIZE"
else
  NPROC="$(cd "$ROOT" && "$PYTHON" - "$CONFIG" <<'RANKS'
import sys

from protactrain.config import load_config

ddp = load_config(sys.argv[1])["grpo"].get("ddp") or {}
print(int(ddp.get("world_size", 1) or 1) if ddp.get("enabled") else 1)
RANKS
  )" || die "could not read grpo.ddp.world_size from $CONFIG"
fi

case "$NPROC" in
  ''|*[!0-9]*) die "rank count must be a positive integer, not '$NPROC'" ;;
esac
[ "$NPROC" -ge 1 ] || die "rank count must be at least 1"

# torchrun sets WORLD_SIZE, RANK and LOCAL_RANK in each child itself. Leaving
# ours in the environment is how a run ends up with a rank disagreeing with
# its launcher, so it is removed once it has been read.
unset WORLD_SIZE

# ------------------------------------------------------- threads, per rank
#
# Now that the rank count is known, divide the job's budget by it. Each rank
# is a full copy of the stack with its own OpenMP pool, so two ranks at the
# full budget ask the container for twice what one rank does -- against the
# same cap.
PER_RANK=$(( THREAD_BUDGET / NPROC ))
[ "$PER_RANK" -ge 1 ] || PER_RANK=1

export PROTACTRAIN_NUM_THREADS="$PER_RANK"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$PER_RANK}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PER_RANK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$PER_RANK}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$PER_RANK}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$PER_RANK}"

VISIBLE="unknown (nvidia-smi not on PATH)"
if command -v nvidia-smi >/dev/null 2>&1; then
  VISIBLE="$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | paste -sd'; ' -)"
  [ -n "$VISIBLE" ] || VISIBLE="none"
fi

# ----------------------------------------------------------------- the command
#
# `-m protactrain.<stage>` is run from the repo root because protactrain
# imports moltrain and protacbench as siblings, and none of the three is
# installed -- they are found on sys.path by being next to each other.
CMD=(
  torchrun
  --standalone            # single node; no rendezvous backend to stand up
  --nnodes=1
  --nproc_per_node="$NPROC"
  -m "protactrain.${STAGE}"
  train
  --config "$CONFIG"
)
CMD+=("${EXTRA[@]+"${EXTRA[@]}"}")

printf '\n==> protactrain %s on %s process(es)\n' "$STAGE" "$NPROC"
printf '    root     %s\n' "$ROOT"
printf '    config   %s\n' "$CONFIG"
printf '    threads  %s per rank (budget %s across %s)\n' \
  "$PER_RANK" "$THREAD_BUDGET" "$NPROC"
printf '    gpus     %s\n' "$VISIBLE"
printf '    command  '
# printf %q so the echoed line is one that can be pasted back into a shell
# unchanged, quoting and all.
printf '%q ' "${CMD[@]}"
printf '\n\n'

if [ "$DRY_RUN" -eq 1 ]; then
  printf '    --dry-run: nothing was started\n\n'
  exit 0
fi

cd "$ROOT"
exec "${CMD[@]}"
