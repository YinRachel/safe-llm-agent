#!/usr/bin/env bash
# LOGIN NODE ONLY (compute nodes are offline). Downloads everything the jobs need into scratch, then builds the env.
# Usage: SCRATCH=/mnt/scratch/users/$USER bash scripts/viking/download.sh   (needs `huggingface-cli login` for the gated Mistral repo)
set -euo pipefail
SCRATCH="${SCRATCH:?set SCRATCH=/mnt/scratch/users/<username>}"
export HF_HOME="$SCRATCH/hf_cache" TRAINSET_ROOT="$SCRATCH/agentft"
mkdir -p "$HF_HOME" "$TRAINSET_ROOT/raw"
ENV_DIR="${ENV_DIR:-$HOME/envs/agentft}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  module purge; module load Python/3.12.3-GCCcore-13.3.0
  python -m venv "$ENV_DIR"
  "$ENV_DIR/bin/pip" install -U pip
  "$ENV_DIR/bin/pip" install -r requirements-trainset.txt vllm
fi
PY="$ENV_DIR/bin/python"
# Raw datasets + eval sets + pins (build_trainset.py download needs raw files in $TRAINSET_ROOT/raw; HF fetches them if missing).
"$PY" -I scripts/build_trainset.py download
# Models: tokenizer, embedding model, generator (123B BF16 checkpoint is ~250 GB; scratch has 512 GB).
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
for repo, rev in (("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
                  ("sentence-transformers/all-MiniLM-L6-v2", None),
                  ("mistralai/Mistral-Large-Instruct-2411", None)):
    path = snapshot_download(repo, revision=rev)
    print(repo, "->", path)  # the snapshot directory name is the commit hash: export it as MISTRAL_REVISION
PY
