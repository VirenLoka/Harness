#!/usr/bin/env bash
#
# Fetch the generative models this repo drives but does not contain.
#
# Pro-GAT and DiffLinker are upstream repositories, not our code, and both
# publish their weights inside their own git history -- Pro-GAT carries all
# 244 MB of checkpoints and datasets as ordinary tracked files. So cloning
# them at a pinned commit gets the code and the weights in one step, and
# nothing large has to live in this repository or in its LFS quota.
#
# That is why `.gitignore` excludes both directories. They are dependencies
# fetched by this script, not vendored source.
#
# DiffLinker is the exception that needs a second step: its own repo has no
# checkpoints, they live on Zenodo, so they are downloaded separately below.
#
# Usage:
#   scripts/setup_models.sh              # everything
#   scripts/setup_models.sh --code-only  # skip the Zenodo downloads
#
# Safe to re-run. An existing clone is fetched and moved to the pinned
# commit; a checkout with local changes is left alone and reported.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Pinned so a later upstream commit cannot silently change what the pipeline
# samples from. Bump deliberately, and re-run the assessment when you do.
PROGAT_URL="https://github.com/drugparadigm/Pro-GAT.git"
PROGAT_REF="d527ea8"

DIFFLINKER_URL="https://github.com/igashov/DiffLinker.git"
DIFFLINKER_REF="fafbe47"

# REINVENT is the ADC linker generator; Linker-GPT is cloned only for its
# curated reagent set, which is what adclinker's scoring is calibrated against.
REINVENT_URL="https://github.com/MolecularAI/REINVENT4.git"
REINVENT_REF="3cf206e"

LINKERGPT_URL="https://github.com/su-group/Linker-GPT.git"
LINKERGPT_REF="d0f2a99"

# REINVENT deleted its priors from the repo and moved them to Zenodo, but the
# blobs are still in its history, so a full clone already has them.
REINVENT_PRIORS_DELETED_AT="1561377efd74e7954fb4a87f88a4b81a0eb6e208"

CODE_ONLY=0
[ "${1:-}" = "--code-only" ] && CODE_ONLY=1

say()  { printf '\n==> %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*" >&2; }

# ---------------------------------------------------------------- the clones

clone_at() {
  local dir="$1" url="$2" ref="$3"

  if [ -d "$dir/.git" ]; then
    if [ -n "$(git -C "$dir" status --porcelain)" ]; then
      warn "$dir has local changes; leaving it as it is"
      warn "  commit or stash them, then re-run to move to $ref"
      return 0
    fi
    say "updating $dir to $ref"
    git -C "$dir" fetch --quiet origin
  elif [ -e "$dir" ]; then
    warn "$dir exists but is not a git checkout; leaving it alone"
    return 0
  else
    say "cloning $dir from $url"
    git clone --quiet "$url" "$dir"
  fi

  git -C "$dir" checkout --quiet "$ref"
  printf '  at %s\n' "$(git -C "$dir" log --oneline -1)"
}

clone_at Pro-GAT     "$PROGAT_URL"     "$PROGAT_REF"
clone_at DiffLinker  "$DIFFLINKER_URL" "$DIFFLINKER_REF"
clone_at REINVENT4   "$REINVENT_URL"   "$REINVENT_REF"
clone_at Linker-GPT  "$LINKERGPT_URL"  "$LINKERGPT_REF"

# ------------------------------------------------ REINVENT's LinkInvent prior

# No download needed: the file was removed from the working tree but its blob
# is still reachable in the clone's own history, and it is checksum-verified
# against the sha512.sum that was deleted alongside it.
say "REINVENT LinkInvent prior"
if [ -s REINVENT4/priors/linkinvent.prior ]; then
  printf '  have REINVENT4/priors/linkinvent.prior\n'
elif [ -d REINVENT4/.git ]; then
  mkdir -p REINVENT4/priors
  if git -C REINVENT4 cat-file -e "${REINVENT_PRIORS_DELETED_AT}~1:priors/linkinvent.prior" 2>/dev/null; then
    printf '  recovering from git history (90 MB)\n'
    git -C REINVENT4 show "${REINVENT_PRIORS_DELETED_AT}~1:priors/linkinvent.prior" \
      > REINVENT4/priors/linkinvent.prior
    git -C REINVENT4 show "${REINVENT_PRIORS_DELETED_AT}~1:priors/sha512.sum" \
      > REINVENT4/priors/sha512.sum 2>/dev/null || true
    if command -v shasum >/dev/null 2>&1 && [ -s REINVENT4/priors/sha512.sum ]; then
      ( cd REINVENT4/priors && shasum -a 512 -c sha512.sum 2>/dev/null | grep -i linkinvent ) \
        || warn "checksum did not verify; re-download from https://doi.org/10.5281/zenodo.15641296"
    fi
  else
    warn "the prior is not in this clone's history (shallow clone?);"
    warn "  download it from https://doi.org/10.5281/zenodo.15641296"
  fi
fi

# ------------------------------------------------- DiffLinker's own weights

# DiffLinker keeps no checkpoints in git; they are Zenodo artefacts. The
# anchor-free GEOM model is the one to use for a two-fragment link, and the
# size GNN lets the linker length be predicted rather than fixed.
fetch() {
  local dest="$1" url="$2"
  if [ -s "$dest" ]; then
    printf '  have %s\n' "$dest"
    return 0
  fi
  printf '  fetching %s\n' "$dest"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dest.part"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$dest.part"
  else
    warn "neither curl nor wget is available; cannot fetch $dest"
    rm -f "$dest.part"
    return 1
  fi
  mv "$dest.part" "$dest"
}

if [ "$CODE_ONLY" -eq 0 ]; then
  say "DiffLinker checkpoints"
  mkdir -p DiffLinker/models
  Z="https://zenodo.org/record/7121300/files"
  Z2="https://zenodo.org/records/10988017/files"

  fetch DiffLinker/models/geom_difflinker.ckpt "$Z/geom_difflinker.ckpt?download=1"
  fetch DiffLinker/models/geom_size_gnn.ckpt   "$Z/geom_size_gnn.ckpt?download=1"
  fetch DiffLinker/models/pockets_difflinker_full.ckpt \
    "$Z2/pockets_difflinker_full_no_anchors_fc_pdb_excluded.ckpt?download=1"
  fetch DiffLinker/models/pockets_difflinker_full_given_anchors.ckpt \
    "$Z2/pockets_difflinker_full_fc_pdb_excluded.ckpt?download=1"
else
  say "skipping Zenodo downloads (--code-only)"
fi

# ------------------------------------------------------------- what is here

say "what is on disk"

check() {
  if [ -s "$1" ]; then
    printf '  ok      %-62s %s\n' "$1" "$(du -h "$1" | cut -f1)"
  else
    printf '  MISSING %s\n' "$1"
  fi
}

check Pro-GAT/checkpoints/protacs_best.ckpt
check Pro-GAT/repair_model/ckpt/best_repair_model.pt
check Pro-GAT/datasets/protacs_test.pt
check DiffLinker/models/geom_difflinker.ckpt
check DiffLinker/models/geom_size_gnn.ckpt
check DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt
check REINVENT4/priors/linkinvent.prior
check Linker-GPT/data/linkers.csv

say "next"
cat <<'EOF'
  python -m protacbench probe     # what this host can do
  python -m protacbench sample    # Pro-GAT sampling          (needs a GPU)
  python -m protacbench assess    # measure, rank, triage     (no GPU)
  python -m protacbench repair    # LLM repair via the harness (needs a model)

  python -m adclinker probe       # what this host can do
  python -m adclinker profile     # measure real ADC linkers    (no GPU)
  python -m adclinker generate    # sample, score, rank         (needs REINVENT)
EOF
