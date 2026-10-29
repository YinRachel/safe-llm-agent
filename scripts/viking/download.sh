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
"$PY" -I scripts/data/build_trainset.py download
# Models: tokenizer, Qwen3-4B-Base (Safe LoRA), Qwen2.5-3B-Instruct (second base model), embedding model, generator. The generator is served in Mistral format (consolidated weights), so
# the duplicate HF-format shards are skipped (the repo holds both: ~490 GB for Mistral-Large instead of ~246 GB).
# GEN_MODEL defaults to Mistral-Large; e.g. GEN_MODEL=mistralai/Ministral-8B-Instruct-2410 for the 8B (~16 GB).
GEN_MODEL="${GEN_MODEL:-mistralai/Mistral-Large-Instruct-2411}" "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo, rev, patterns in (("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", None),
                            # unaligned reference for Safe LoRA (posthoc_defense.py --method safelora)
                            ("Qwen/Qwen3-4B-Base", "906bfd4b4dc7f14ee4320094d8b41684abff8539", None),
                            # second base model (scripts/viking/qwen25_3b.sh)
                            ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1", None),
                            ("sentence-transformers/all-MiniLM-L6-v2", None, None),
                            (os.environ["GEN_MODEL"], None, ["consolidated*", "params.json", "tekken.json", "tokenizer.model*"]  # Ministral ships tekken.json, Mistral-Large-2411 tokenizer.model.v7)):
    path = snapshot_download(repo, revision=rev, allow_patterns=patterns)
    print(repo, "->", path)  # the snapshot directory name is the commit hash: export it as MISTRAL_REVISION
PY
