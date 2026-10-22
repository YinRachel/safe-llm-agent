# Benign baseline: Qwen3-4B-Instruct-2507

Evaluate the untouched instruction-tuned checkpoint before fine-tuning or applying defenses.
The target-model runner currently exposes only benign evaluation. A separate local classifier can score saved completions for a later harmful-task study stage.

## Benchmark scope

- **AgentHarm:** uses the official `inspect_evals.agentharm.agentharm_benign` task, its mock tools, agent loop, and grading functions. `--split test_public` selects the public benign task data; `--split val` is for debugging. Do not replace this with single-turn prompt generation.
- **HarmBench:** the [linked walledai dataset](https://huggingface.co/datasets/walledai/HarmBench) documents harmful behavior prompts and does not document a benign split. `evaluate.py harmbench` exits before contacting any model. There is no scientifically valid HarmBench benign result to report from that release. The repository is also gated on Hugging Face; this workflow does not attempt to bypass access conditions.
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

The separate `scripts/score_harmbench.py` scores **already saved completions**. It does not generate harmful tasks with Qwen, call Runware, or load `.env`. Standard and contextual text behaviors are supported; copyright/hash and multimodal cases are rejected with a message to use the full upstream evaluator. The original full benchmark also uses specialized copyright hash checks, so not every HarmBench behavior should be routed to the language-model classifier.

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

The classifier reads raw official prompts from the checkout's `eval_utils.py` without executing the file, clips completions to the first 512 classifier-tokenizer tokens, and generates one yes/no token greedily. It uses no additional chat-template wrapping. Outputs include original completion fields, clipped generation, raw classifier answer, and label. Unknown IDs, empty sample lists, unsupported tasks, and invalid labels cannot silently become successful scores. ASR is the mean of per-behavior success fractions over the supplied IDs. A subset is only a subset score; preserve its ID set and do not report it as a full benchmark result.

The 13B classifier's unquantized weights need roughly 26 GB plus runtime memory. Use a 40/48 GB GPU or `--tensor-parallel-size 2` across suitable GPUs, and free the Qwen serving memory first if sharing hardware. Download/cache the classifier once, then use `--offline` for scoring without Hugging Face network access. For immutable runs supply `--revision <HF-commit>`. `--classifier /local/model/path` is also supported. Initial downloads require internet; no inference data goes to an external judge. Save the resolved serving/classifier environment using `pip freeze` after GPU validation.

`results.json` preserves grouped labels, `predictions.jsonl` records each classified completion immediately, and `summary.json` reports sample counts and ASR. Invalid outputs have label `-1`, suppress the aggregate ASR, and fail the run. The manifest records classifier identity, template literals, upstream commit (when available), file hashes, and package versions. This is a vLLM-backed implementation of the upstream classifier scoring protocol, not a claim of full HarmBench pipeline parity or benign helpfulness measurement.

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
