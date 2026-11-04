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
export PROTACTRAIN_NUM_THREADS="${PROTACTRAIN_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$PROTACTRAIN_NUM_THREADS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PROTACTRAIN_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$PROTACTRAIN_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$PROTACTRAIN_NUM_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$PROTACTRAIN_NUM_THREADS}"

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
# grpo.ddp.world_size in the config is 2 and so is the default here. WORLD_SIZE
# is read because a smoke test on one card is the first thing anyone wants:
# `WORLD_SIZE=1 scripts/launch_ddp.sh grpo`.
NPROC="${WORLD_SIZE:-2}"
case "$NPROC" in
  ''|*[!0-9]*) die "WORLD_SIZE must be a positive integer, not '$NPROC'" ;;
esac
[ "$NPROC" -ge 1 ] || die "WORLD_SIZE must be at least 1"

# torchrun sets WORLD_SIZE, RANK and LOCAL_RANK in each child itself. Leaving
# ours in the environment is how a run ends up with a rank disagreeing with
# its launcher, so it is removed once it has been read.
unset WORLD_SIZE

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
printf '    threads  %s\n' "$PROTACTRAIN_NUM_THREADS"
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
