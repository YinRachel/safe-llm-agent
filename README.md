# Benign baseline: Qwen3-4B-Instruct-2507

Evaluate the untouched instruction-tuned checkpoint before fine-tuning or applying defenses.
AgentHarm remains a benign baseline. The separate HarmBench pipeline now supports harmful direct requests over the full official text CSV, followed by local scoring.

## Benchmark scope

- **AgentHarm:** uses the official `inspect_evals.agentharm.agentharm_benign` task, its mock tools, agent loop, and grading functions. `--split test_public` selects the public benign task data; `--split val` is for debugging. Do not replace this with single-turn prompt generation.
- **HarmBench:** the [linked walledai dataset](https://huggingface.co/datasets/walledai/HarmBench) documents harmful behavior prompts and does not document a benign split. `evaluate.py harmbench` exits before contacting any model. There is no scientifically valid HarmBench benign result to report from that release. For harmful direct-request evaluation, use `scripts/viking_harmbench.sbatch` as described below. The repository is also gated on Hugging Face; this workflow does not attempt to bypass access conditions.
- **BFCL v3 multi-turn:** the Berkeley Function-Calling Leaderboard multi-turn categories (`multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`; 800 samples) through `inspect_evals.bfcl`. Scoring is deterministic (final backend state and execution results); no judge is needed.
- **τ-bench:** the maintained [τ²-bench](https://github.com/sierra-research/tau2-bench) through `inspect_evals.tau2`. Its `retail` and `airline` domains are the original τ-bench domains (with upstream's task fixes); `telecom` and `banking` are optional. The evaluated Qwen model is the customer-service agent, and a **separate** model plays the simulated user. Scoring is deterministic (database state/actions), with no LLM judge.
- **Optional benign chat:** a separate JSONL generation runner accepts manually reviewed benign prompts. These outputs are labeled `custom_benign_chat`, not HarmBench. The supplied two examples are transport smoke tests, not a research benchmark. Choose and freeze a real benign chat evaluation set before reporting chat utility results.

Sources checked 2026-10-22:
[AgentHarm dataset card](https://huggingface.co/datasets/ai-safety-institute/AgentHarm),
[current official implementation](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentharm),
[HarmBench repository](https://github.com/centerforaisafety/HarmBench),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/).
The AgentHarm HF card shows historical CLI commands; this project uses the current Inspect Evals API instead of mixing the historical Inspect version with modern Qwen serving.

## GPU server setup

Use Linux, Python 3.11 or 3.12, an NVIDIA GPU, and a CUDA/driver combination supported by your vLLM installation. A 24 GB GPU is a useful starting configuration for this 4B checkpoint; actual memory needs depend on context length and concurrent sequences. BF16/FP16 weights alone occupy about 8 GB. Reduce context/concurrency if memory is tight. No quantization is enabled by this project.

Create separate serving and evaluation environments to keep vLLM's CUDA dependencies separate:

```bash
python3 -m venv .venv-serve
.venv-serve/bin/pip install vllm

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

In terminal 1, activate the serving environment and launch the model:

```bash
source .venv-serve/bin/activate
# For final runs, set MODEL_REVISION to the HF model's immutable commit hash.
# export MODEL_REVISION=<commit>
bash scripts/serve_qwen.sh
```

The script uses the checkpoint's chat template and Hermes tool parsing, with automatic tool choice. This Instruct-2507 checkpoint is a non-thinking model; no reasoning parser is enabled. The server binds to localhost. Run evaluation on the same host, or access it through an SSH tunnel. Optional environment settings: `MODEL_PATH` (HF model or local checkpoint), `SERVED_MODEL_NAME`, `PORT`, `MAX_MODEL_LEN`, `TENSOR_PARALLEL_SIZE`, `GPU_MEMORY_UTILIZATION`. Additional vLLM flags can be passed to the shell script. The default is a 32K context cap, not the checkpoint's full advertised context length.

In terminal 2:

```bash
source .venv/bin/activate
python scripts/evaluate.py preflight
```

Preflight checks `/models`, a benign chat completion, and a harmless automatic weather tool call. It never executes that tool. A failure must be fixed before interpreting agent scores.

## Run AgentHarm benign

The official scorer combines Python grading functions with semantic checks and refusal judgments. Both LLM roles share the configured judge; the Python grading functions remain the official implementation.

This project's `.env` is now configured for the user-selected **GPT-6 Luna through Runware**, using the documented AIR ID `openai:gpt@6-luna`. The key stays in `.env` and is loaded automatically from the project root; shell environment values take precedence. The selected model is an alternative judge, so these results are not a reproduction of the original GPT-4o judging protocol. A preflight verifies transport and expected answer formats, not agreement with human judgments.

Equivalent configuration (see `.env.example`; never commit your key):

```dotenv
RUNWARE_API_KEY=your-key
JUDGE_PROVIDER=runware
RUNWARE_JUDGE_MODEL=openai:gpt@6-luna
```

Run a judge-only connection and format check before starting the GPU experiment:

```bash
python scripts/evaluate.py judge-preflight
```

This makes three small external API calls using the same Inspect transport as evaluation, testing refusal/non-refusal and pass/fail output. Full AgentHarm runs repeat this check automatically. Runware receives the judge prompts/transcripts and bills inference. Its endpoint is `https://api.runware.ai/v1`; the solver continues using the independent localhost vLLM endpoint. No model is silently substituted if the configured model fails. Live access to the selected model has not been verified in development.

To use the benchmark's original GPT-4o snapshot directly, set `OPENAI_API_KEY` and select the provider explicitly:

```bash
python scripts/evaluate.py judge-preflight --judge-provider openai
python scripts/evaluate.py agentharm --judge-provider openai --output results/base-gpt4o-judge
```

That uses `openai/gpt-4o-2024-08-06`, subject to provider availability. A Runware key cannot authenticate to OpenAI's own endpoint. The current Runware catalog does not document GPT-4o, so this project does not invent a Runware GPT-4o ID. Provider information: [Runware compatible APIs and model IDs](https://runware.ai/docs/tools/compatible-apis).

First run a small validation subset:

```bash
python scripts/evaluate.py agentharm --split val --limit 2 --output results/val-smoke
```

Then run the full public benign task:

```bash
python scripts/evaluate.py agentharm --output results/qwen-base-agentharm-benign
```

For a separately hosted local judge, use a distinct server/model, for example:

```bash
python scripts/evaluate.py agentharm \
  --judge-provider compatible \
  --judge openai-api/judge/your-judge-model \
  --judge-base-url http://127.0.0.1:8001/v1 \
  --output results/qwen-base-local-judge
```

Inspect compatible-model identifiers have a service prefix: `openai-api/judge/your-judge-model` sends `your-judge-model` to the server.

`JUDGE_API_KEY` sets credentials for an explicit judge endpoint; `TARGET_API_KEY` is for the Qwen endpoint. Both default to a placeholder for unauthenticated local servers. An alternative judge changes the evaluation protocol: validate it and report its identity; do not treat it as directly equivalent to the paper's judge. Avoid self-grading with the evaluated 4B model.

Common options: `--model`, `--base-url`, `--max-tokens` (per generation, default 4096), `--concurrency` (default 4), `--seed` (default 42). Generation temperature is fixed to 0 for this deterministic baseline protocol, which should be stated in reports. Inspect Evals supplies the official task's agent budget/defaults. Review token/context exhaustion in the logs before adopting the budget for the full study.

Each output directory must be new. Artifacts:

- `manifest.json`: settings, UTC timestamp, installed package versions, code commit when available, and declared model revision.
- `dataset.jsonl` and `dataset.sha256`: the exact loaded benign samples and their checksum (including samples beyond a smoke-test limit).
- `inspect/*.eval`: full samples, tool trajectories, grading breakdowns, judge outputs, and errors.
- `summary.json`: native aggregate metrics and completion status.
- `failure.json`: failure diagnostic, if the run raises an exception.

Inspect logs with `inspect view --log-dir results/qwen-base-agentharm-benign/inspect`.

On **benign** tasks, `avg_score` measures mean rubric completion, `avg_full_score` measures the fraction fully completed, and `avg_refusals` measures refusal rate. Higher completion and lower refusal are desirable. Preserve category results and sample counts. These are benign utility/over-refusal results, not harmful-task attack success rates. Inspect sample-level judge explanations for errors and malformed judge answers; upstream refusal scoring can swallow judge errors. The runner marks a run invalid if recorded refusal explanations contain these errors, saving `judge_errors.json`; it does not reclassify API failures as successful non-refusals. Review semantic judge outputs as well, since upstream semantic grading warns on malformed pass/fail answers.

## Agent capability benchmarks: BFCL v3 multi-turn and τ-bench

Both are benign tool-use capability benchmarks and use the same vLLM/Qwen endpoint, Inspect transport, run-directory rules and temperature 0 / seed protocol as AgentHarm. Run them on the same checkpoints before and after fine-tuning or defenses to measure capability retention. The only added dependency is `mpmath`, listed explicitly in `requirements.txt` for BFCL's math backend; install it with `pip install "mpmath>=1.3,<2"` if you don't want to re-resolve the whole file.

```bash
# Smoke tests first
python scripts/evaluate.py bfcl-multiturn --limit 5 --output results/bfcl-smoke
python scripts/evaluate.py tau-bench --domains retail --limit 2 --message-limit 20 --output results/tau-smoke
# Full runs
python scripts/evaluate.py bfcl-multiturn --output results/qwen-base-bfcl-multiturn
python scripts/evaluate.py tau-bench --output results/qwen-base-tau-bench
```

**BFCL options:** `--categories` takes any of the four leaderboard categories above or `multi_turn_composite` (default: the four leaderboard categories). The dataset and backend source are downloaded from upstream's GitHub repository at the commit pinned by `inspect_evals`, cached in Inspect Evals' cache directory, and the commit is recorded as `bfcl_data_commit` in `manifest.json`. `--limit N` takes the first N samples in sorted ID order, so a small smoke test only covers `multi_turn_base`. The V4 memory/web-search categories and single-turn categories are intentionally excluded. Summary metrics include overall accuracy and one accuracy per category.

**τ-bench options:** `--domains` takes `retail airline telecom banking` (default `retail airline`); `--message-limit` caps messages per sample (upstream default 100). Samples run in upstream's dataset order (`shuffle=False`) so `--limit` is reproducible. The user simulator reuses the judge routing (`--judge-provider`, `--judge`, `--judge-base-url`, or the `.env` Runware settings), including the Runware `openai:gpt@6-luna` default from `.env`; `manifest.json` records it as `user_simulator`. Never use the evaluated 4B model as its own user simulator: a weak simulator lowers scores for reasons unrelated to the agent. A simulator preflight runs before the GPU endpoint is exercised, and changing the simulator changes the protocol, so fix and report it across all compared checkpoints. The reported metric is single-trial accuracy (pass^1); the pass^k reliability metric from the paper is not computed. τ-bench is token-heavy (upstream reports tens of millions of tokens per domain for small hosted models), so run `--limit` smoke tests first.

**Context length:** BFCL `multi_turn_long_context` and τ-bench conversations grow far beyond the 32K serving default in `scripts/serve_qwen.sh`. Serve with a larger window, for example `MAX_MODEL_LEN=65536 bash scripts/serve_qwen.sh` (the Viking script below does this by default). Overflowing samples end early and score as failures, so `summary.json` lists them as `context_exhausted_sample_ids` for review; treat many of these as a serving-configuration problem, not model capability.

Artifacts follow the AgentHarm layout: `manifest.json`, `dataset.jsonl` (each sample tagged with its `task`) and `dataset.sha256`, `inspect/*.eval`, `summary.json` (one entry per task with `valid`, native metrics, `errored_sample_ids`, `context_exhausted_sample_ids`) and `failure.json` on exceptions. A run is invalid and exits with an error if any task is incomplete or has sample errors. Upstream's τ-bench loop catches per-turn exceptions and prints them instead of failing the sample, so check the console output and Inspect logs for `Error in user_simulator_agent loop` before trusting scores.

Viking: compute nodes need the BFCL data in the Inspect Evals cache and (for τ-bench) outbound access to the user-simulator endpoint. Warm the BFCL cache on a login node once with `python -c "from inspect_evals.bfcl import bfcl; bfcl(categories=['multi_turn_base','multi_turn_miss_func','multi_turn_miss_param','multi_turn_long_context'])"`, then:

```bash
sbatch --account=YOUR_PROJECT_CODE scripts/viking_agent_benchmarks.sbatch bfcl-multiturn
sbatch --account=YOUR_PROJECT_CODE scripts/viking_agent_benchmarks.sbatch tau-bench --domains retail airline
```

The first argument selects the benchmark; the remaining arguments go to `evaluate.py`. The job requests 12 hours (override with `sbatch --time`), serves Qwen with a 64K context (override with `MAX_MODEL_LEN`), and writes `results/<benchmark>-JOBID/`.

References: [BFCL V3 multi-turn](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html), [official BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard), [τ-bench paper](https://arxiv.org/abs/2406.12045), [τ²-bench paper](https://arxiv.org/abs/2506.07982), [Inspect Evals BFCL and Tau2 ports](https://github.com/UKGovernmentBEIS/inspect_evals).

## Optional custom benign chat generation

```bash
python scripts/evaluate.py chat --data data/benign_chat.example.jsonl \
  --output results/chat-smoke
```

Each JSONL row requires a unique string `id`, nonempty `prompt` and `source`, and boolean `benign: true`. That flag records human review; it is not an automatic content safety classifier. The entire input file is validated before inference. The runner records its SHA256 and flushes each generated response to `responses.jsonl`, including finish reason and token use. Truncated responses are counted. `summary.json` explicitly marks outputs as unscored: assessing helpfulness/correctness requires human review or a validated benign utility rubric. No refusal keyword heuristic is presented as a benchmark score.

## Reproducibility and next research stages

Save immutable model and dataset revisions, both environments' `pip freeze` output, GPU/driver details, and the full run directories before comparing checkpoints. The supplied `requirements.lock.txt` freezes the evaluation environment used for import and offline integration checks (Python 3.12/Linux); `requirements.txt` describes direct dependency ranges. GPU serving must be validated and frozen separately on your server. Inspect controls AgentHarm downloads/caching; the runner snapshots loaded benign samples and their checksum. Archive the cache as well when reproducing a published experiment. Set the same model revision when starting vLLM and running evaluation; the evaluator records the environment value but cannot verify that a running server loaded that revision. Seed and temperature do not guarantee bitwise reproducibility across hardware or serving versions.

Keep benchmark examples evaluation-only, particularly AgentHarm's explicit prohibition on training with its examples. Fix the data, prompting, tool configuration, budgets, judge, and evaluation software across the base/fine-tuned/defended comparisons. Benign evaluation alone cannot establish restored safety; harmful-task evaluation is a separate later stage.

Local offline checks (no GPU, dataset downloads, or judge calls):

```bash
.venv/bin/python -m unittest discover -s tests -v
bash -n scripts/serve_qwen.sh
```

## HarmBench local classifier configuration

HarmBench's text classifier is a fine-tuned causal language model, **`cais/HarmBench-Llama-2-13b-cls`**. It needs no external GPT judge. Its `yes` label means the completion exhibits the specified harmful behavior. A `no` label does not establish that the model explicitly refused, and neither label measures benign utility. This leaves the benign-only dataset restriction unchanged.

The separate `scripts/score_harmbench.py` scores **already saved completions**. Generation is now handled by `scripts/generate_harmbench.py`. The scorer does not call Runware or load `.env`. Standard and contextual text behaviors use the local classifier. Add `--include-hash-checks` to score copyright cases with the official hash grader and its reference files; this also requires spaCy, its `en_core_web_sm` model, and datasketch. Multimodal cases remain outside this text-only workflow. The original full benchmark also uses specialized copyright hash checks, so not every HarmBench behavior should be routed to the language-model classifier.

On the GPU server, use a separate environment and an official checkout for exact prompt templates:

```bash
python3 -m venv .venv-harmbench
.venv-harmbench/bin/pip install -r requirements-harmbench.txt
git clone https://github.com/centerforaisafety/HarmBench.git /path/to/HarmBench
# Check out your chosen immutable upstream commit and record it before final runs.
```

Then score standard/contextual text completions:

```bash
.venv-harmbench/bin/python scripts/score_harmbench.py \
  --harmbench-repo /path/to/HarmBench \
  --behaviors /path/to/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --completions /path/to/completions.json \
  --output results/harmbench-classifier
```

The CSV must preserve `BehaviorID`, `Behavior`, `Tags` and `ContextString` for contextual tasks. Completion JSON uses the upstream schema:

```json
{"behavior-id-from-csv": [{"test_case": "original prompt", "generation": "saved model response"}]}
```

The classifier reads raw official prompts from the checkout's `eval_utils.py` without executing the file, clips completions to the first 512 classifier-tokenizer tokens, and generates one yes/no token greedily. It uses no additional chat-template wrapping. Outputs include original completion fields, clipped generation, raw classifier answer, and label. Unknown IDs, empty sample lists, unsupported tasks, and invalid labels cannot silently become successful scores. `--require-full-coverage` additionally rejects completion files missing any CSV behavior. ASR is the mean of per-behavior success fractions over the supplied IDs. A subset is only a subset score; preserve its ID set and do not report it as a full benchmark result.

The 13B classifier's unquantized weights need roughly 26 GB plus runtime memory. Use a 40/48 GB GPU or `--tensor-parallel-size 2` across suitable GPUs, and free the Qwen serving memory first if sharing hardware. Download/cache the classifier once, then use `--offline` for scoring without Hugging Face network access. For immutable runs supply `--revision <HF-commit>`. `--classifier /local/model/path` is also supported. Initial downloads require internet; no inference data goes to an external judge. Save the resolved serving/classifier environment using `pip freeze` after GPU validation.

`results.json` preserves grouped labels, `predictions.jsonl` records each classified completion immediately, and `summary.json` reports sample counts and ASR. Invalid outputs have label `-1`, suppress the aggregate ASR, and fail the run. The manifest records classifier identity, template literals, upstream commit (when available), file hashes, and package versions. This is a direct-request text baseline using upstream classifier prompts and optional upstream copyright scoring. It does not run every HarmBench attack method or measure benign helpfulness.

References: [official classifier](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls), [upstream evaluation](https://github.com/centerforaisafety/HarmBench/blob/main/evaluate_completions.py), [classifier templates and labeling](https://github.com/centerforaisafety/HarmBench/blob/main/eval_utils.py).

## York Viking batch submission

Create the evaluation and serving environments on Viking itself using the setup above; copying a local virtual environment is not portable. Load the Python/CUDA modules appropriate for those environments in `scripts/viking_agentharm.sbatch` if your installation needs them. Submit from the project root:

```bash
# First validate two benign samples.
sbatch --account=YOUR_PROJECT_CODE scripts/viking_agentharm.sbatch --split val --limit 2
# Then the full public benign AgentHarm task.
sbatch --account=YOUR_PROJECT_CODE scripts/viking_agentharm.sbatch
squeue -u "$USER"
```

This requests one GPU on Viking's documented `gpu` partition, starts the Qwen server on the allocated compute node, waits for health readiness, runs evaluation against that node's localhost endpoint, and stops the server on exit. No browser, manually opened localhost session, or second allocation is needed. Use a real project account in place of `YOUR_PROJECT_CODE`. Logs are `qwen-agentharm-JOBID.out` and `logs/vllm-JOBID.log`; results are `results/agentharm-JOBID/`. Adjust the requested time and memory to your measured workload. `gpuplus` is an optional partition override for an H100 allocation.

The judge requires outbound access to Runware from the compute node, and initial model/dataset downloads require access to Hugging Face. Test the judge first; if the cluster blocks outbound inference access, a separate permitted judge routing arrangement is needed. Do not run GPU inference on a login node. The local HarmBench classifier is a separate job over existing completion files, as described above; it does not produce benign benchmark tasks.

Viking references: [GPU job examples](https://vikingdocs.york.ac.uk/using_viking/jobscript_examples.html), [resource partitions](https://vikingdocs.york.ac.uk/using_viking/resource_partitions.html).

### Installation troubleshooting on Viking

Check `.venv/bin/python --version` first and use Python 3.11 or 3.12 for this workflow. The development `requirements.lock.txt` is a record of a different machine, not a Viking-validated installation lock. Resolve `requirements.txt` on Viking, run the offline tests, then save `.venv/bin/python -m pip freeze > requirements.viking.lock.txt` for subsequent experiment reproducibility. A `from versions: none` error can reflect Python compatibility, package-index visibility, or connectivity; it does not alone prove that the package does not exist. If public PyPI is available on your installation, an explicit `--index-url https://pypi.org/simple` can distinguish index configuration issues.

## Full HarmBench direct requests on Viking

This HarmBench path evaluates one direct request per behavior, without a jailbreak or attack search. It defaults to the official upstream `data/behavior_datasets/harmbench_behaviors_text_all.csv`, the full **text** set, including standard, contextual, and copyright behaviors. It is not the multimodal benchmark and not the full suite of attack methods. The gated `walledai/HarmBench` card links to this upstream data; using the CSV preserves behavior IDs, context and tags needed for scoring. This workflow does not claim to load or exactly reproduce the gated Hugging Face repackaging.

Copy the updated scripts and `requirements-harmbench.txt` to Viking, then prepare the existing GPU serving environment once:

```bash
cd ~/safe-llm-agent
module load Python/3.12.3-GCCcore-13.3.0
.venv-serve/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-harmbench.txt
.venv-serve/bin/python -m spacy download en_core_web_sm
mkdir -p third_party
git clone https://github.com/centerforaisafety/HarmBench.git third_party/HarmBench
```

Clone only once. If an official checkout already exists elsewhere, set `export HARMBENCH_REPO=/absolute/path/to/HarmBench` instead. For final research runs, check out an immutable upstream commit and keep it fixed across all checkpoints. The checkout supplies CSV data, classifier templates, Python hash grading code, and reference pickle files; use the official trusted source. Both CSV/template hashes and selected reference-file hashes are saved in the run manifests.

First test **one behavior from each functional category**, exercising both the classifier and copyright hash path:

```bash
sbatch --account=YOUR_PROJECT_CODE --job-name=harmbench-smoke \
  scripts/viking_harmbench.sbatch --smoke
```

For only the first two rows, use `--limit 2` instead; those rows may not exercise every scoring method. Both tests still load the full model and classifier weights, so startup/download time can dominate the small run.

After the smoke run succeeds, submit the full text set with identical generation settings:

```bash
sbatch --account=YOUR_PROJECT_CODE --job-name=harmbench-full \
  scripts/viking_harmbench.sbatch
```

No manual model server or localhost connection is needed. Within one GPU allocation, process 1 loads Qwen and writes completions; it exits before process 2 loads the 13B classifier. The second process uses official Python hash grading for copyright cases and the classifier for other text cases. No Runware/OpenAI judge is involved. The batch script includes the confirmed Viking Python module. It requests one GPU, 64 GB host RAM and six hours; adjust the wall time after measuring a smoke run.

Monitoring and results:

```bash
squeue -u "$USER"
tail -f harmbench-smoke-JOBID.out
cat results/harmbench-JOBID/summary.json
```

`generation/manifest.json` records source count, selection mode, selected IDs, source hash, model identity and generation configuration. `generation/behaviors.csv` freezes selected samples; `generation/responses.jsonl` records responses, finish reasons and token counts immediately; `generation/completions.json` is saved atomically after every batch. `generation/generation_summary.json` counts truncated outputs. `scoring/predictions.jsonl` and `scoring/results.json` preserve raw labels and scoring method. `scoring/summary.json` describes coverage of the selected CSV; the **top-level `summary.json`** also records whether the original source CSV was fully evaluated. A smoke run must never be reported as a full-set score.

The default protocol uses the model's native chat template, one user message per behavior, greedy generation, seed 42, and a 512-token generation cap. Context is formatted exactly like upstream DirectRequest: `ContextString + "\n\n---\n\n" + Behavior`. No system prompt is added. The classifier sees the first 512 classifier-tokenizer tokens of the response, as in the upstream evaluation's default clipping. Supply `--max-tokens` if choosing a different generation budget and keep that setting identical between smoke/full runs and model comparisons. This budget is part of the reported protocol; HarmBench's historical generation script defaults to 256 new tokens.

The full-run command evaluates every row of the supplied CSV: no standard/context/copyright cases are silently dropped. Missing references or invalid labels fail the run. `asr` is the mean harmful-behavior success rate; lower is safer for this direct-request baseline. It is not a refusal rate, and a direct-request score alone is not worst-case adversarial robustness.

This job expects an A40-class GPU with enough memory for the 13B classifier (roughly 26 GB of unquantized weights plus runtime memory). Sequential execution prevents Qwen and the classifier from occupying GPU memory together. Initial downloads still need Hugging Face access. After caching both models, set `HB_OFFLINE=1` before submitting to disable Hugging Face downloads. Use `MODEL_REVISION` and `CLASSIFIER_REVISION` for immutable model revisions. An existing local classifier can be selected with `CLASSIFIER_PATH`; a local Qwen model can be selected with the generation option `--model /path/to/model`.

The pipeline has offline end-to-end tests with fake inference backends, including context construction, complete CSV coverage, and routing copyright cases to Python hash grading. Actual GPU inference must be validated on Viking with the smoke command.

### FlashInfer cannot find nvcc on Viking

If startup fails with `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`, FlashInfer requires a CUDA toolkit for kernel compilation in that job. A working GPU driver alone does not supply the compiler. Check `module avail CUDA` and `.venv-serve/bin/python -c 'import torch; print(torch.version.cuda)'` to identify an appropriate toolkit for the installed PyTorch build and driver. The Viking batch scripts accept `CUDA_MODULE` as an exact module name and check `nvcc` before loading models. Export `CUDA_MODULE=the-exact-available-module` before submission; the job loads it and derives `CUDA_HOME` from the resolved compiler path. Do not create a fictitious `/usr/local/cuda` directory or change generation budgets to address this error.

### CUDA 12.8 environment for the available Viking toolkit

The supplied Viking module list includes `CUDA/12.8.0` and `CUDA/12.9.0`, but not CUDA 13.0. For the observed PyTorch `2.13.0+cu130` / missing compiler failure, create a separate environment instead of mixing CUDA 12.x compilation with the CUDA 13 PyTorch build. `requirements-viking-cu128.txt` selects vLLM 0.10.2, its PyTorch 2.8 stack with CUDA 12.8 wheels, and Transformers 4.55.2. This follows [vLLM 0.10.2's CUDA 12.8 installation documentation](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html); GPU/driver compatibility still needs a Viking smoke run.

```bash
cd ~/safe-llm-agent
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0
python3.12 -m venv .venv-serve-cu128
.venv-serve-cu128/bin/python -m pip install --upgrade pip --index-url https://pypi.org/simple
.venv-serve-cu128/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-viking-cu128.txt
.venv-serve-cu128/bin/python -m spacy download en_core_web_sm
.venv-serve-cu128/bin/python -m pip check
.venv-serve-cu128/bin/python -c 'import torch, vllm; print("vLLM:", vllm.__version__); print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda)'
nvcc --version

export CUDA_MODULE=CUDA/12.8.0
export HB_PYTHON="$PWD/.venv-serve-cu128/bin/python"
export SERVE_ENV="$PWD/.venv-serve-cu128"
sbatch --account=YOUR_PROJECT_CODE --job-name=harmbench-cu128-smoke scripts/viking_harmbench.sbatch --smoke
```

Copy this new requirements file along with `requirements-harmbench.txt` and the updated scripts to Viking before installation. Keep `.venv` for Inspect evaluation; do not install its requirements into this GPU environment. The environment exports must be repeated in a new shell before submitting jobs (or set as explicit paths in the corresponding batch script). They select the new environment for both HarmBench and Qwen serving. After the smoke run succeeds, submit the same HarmBench job without `--smoke` for the full text set. This configuration is documented and pinned at the primary GPU-package level, but has not been GPU-tested by the development agent; save `pip freeze` after validation.

### Classifier CUDA initialization failure after successful generation

When Qwen generation succeeds but the classifier's forked worker fails CUDA initialization, force fresh spawned vLLM workers. The Python entry points and HarmBench batch job now set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before loading native libraries. They also default tokenizer parallelism off. This addresses the known incompatibility between `fork` and previously initialized CUDA/native runtime state; it does not establish that every CUDA initialization error has that cause. See [vLLM multiprocessing documentation](https://docs.vllm.ai/en/v0.10.2/design/multiprocessing.html). GPU confirmation still requires retrying the failed stage.

Copy the updated `scripts/generate_harmbench.py`, `scripts/score_harmbench.py`, and `scripts/viking_harmbench.sbatch` to Viking. Saved generation results can be reused under a new allocation:

```bash
cd ~/safe-llm-agent
export CUDA_MODULE=CUDA/12.8.0
export HB_PYTHON="$PWD/.venv-serve-cu128/bin/python"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
sbatch --account=YOUR_PROJECT_CODE --job-name=harmbench-score-retry \
  scripts/viking_harmbench.sbatch \
  --score-existing results/harmbench-37930626/generation
```

`--score-existing` takes exactly one generation directory containing the saved manifest, selected behavior CSV and completion JSON. It skips Qwen, verifies full completion coverage of the selected CSV, and writes scoring plus the combined summary to the new job's `results/harmbench-JOBID/`. Original artifacts are preserved. If the saved data is a smoke selection, the retry remains a smoke result, not a full-set score. Once scoring succeeds, submit a new full job without `--score-existing` or `--smoke`.

### Runware rejects max_tokens for GPT-6 Luna

Runware's AIR ID `openai:gpt@6-luna` does not match Inspect's standard OpenAI model-name detection. Without a family registration, Inspect sends the legacy `max_tokens` field, which the endpoint rejects. `make_judge()` now registers the AIR service ID with `ModelInfo(family="gpt-6-luna")`, keeping the actual provider model ID unchanged. Inspect then translates each call's token budget to `max_completion_tokens`, including calls inside AgentHarm's official graders. This applies to the preflight and both semantic/refusal roles, and leaves Qwen and other configured judge providers unchanged.

For this judge, `reasoning_effort="none"` is set explicitly so short grading responses do not spend the small completion budget on hidden reasoning. This is a supported [GPT-6 Luna setting](https://developers.openai.com/api/docs/models/gpt-6-luna), and is recorded alongside the family in run metadata. It is a declared alternative-judge configuration, not a reproduction of GPT-4o judging. OpenAI documents `max_completion_tokens` as the successor to the legacy token-budget field in its [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat).

Copy the updated `scripts/evaluate.py` to Viking and rerun `.venv/bin/python scripts/evaluate.py judge-preflight --judge-provider runware` before submitting another GPU allocation. A regression test exercises the real Inspect-to-SDK request path with a mocked API response and checks that only `max_completion_tokens` is sent, the AIR model ID is preserved, and the intended reasoning effort is included. Live acceptance by Runware must still be verified with the preflight.
