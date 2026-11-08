"""scripts/viking/eval_all.sh against a fake sbatch: seven jobs, unchanged scripts, template and model routing."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
FAKE_SBATCH = """#!/usr/bin/env bash
n=$(( $(ls "$LOG" | wc -l) + 1 ))
{ printf 'ARGS'; printf ' %s' "$@"; printf '\\nMODEL_PATH=%s\\nMODEL_REVISION=%s\\nCHAT_TEMPLATE=%s\\n' \\
  "${MODEL_PATH:-}" "${MODEL_REVISION:-<unset>}" "${CHAT_TEMPLATE:-<unset>}";
  printf 'TOOL_CALL_PARSER=%s\\nMOSAIC_SCAFFOLD=%s\\nRESULTS_DIR=%s\\n' \\
  "${TOOL_CALL_PARSER:-<unset>}" "${MOSAIC_SCAFFOLD:-<unset>}" "${RESULTS_DIR:-<unset>}"; } > "$LOG/$n"
echo "10$n"
"""


class EvalAllTests(unittest.TestCase):
    def run_script(self, *args, revision="should-be-unset", extra_env=None):
        self.tmp = tempfile.TemporaryDirectory()
        work = Path(self.tmp.name)
        shutil.copytree(ROOT / "scripts", work / "scripts")
        (work / "templates").mkdir()
        (work / "templates/ptst.jinja").write_text("T")
        for name in ("glm4_tools.jinja", "llama31_tools.jinja", "mosaic.jinja"):
            shutil.copy(ROOT / "templates" / name, work / "templates" / name)
        model = work / "results/qwen4B_LoRA/p10-1/merged"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        bin_dir, log = work / "bin", work / "log"
        bin_dir.mkdir(), log.mkdir()
        (bin_dir / "sbatch").write_text(FAKE_SBATCH)
        (bin_dir / "sbatch").chmod(0o755)
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "LOG": str(log)}
        env.pop("MODEL_REVISION", None)
        env.update(extra_env or {})
        if revision:
            env["MODEL_REVISION"] = revision
        subprocess.run(["bash", "scripts/viking/eval_all.sh", *args], cwd=work, env=env, check=True, capture_output=True)
        calls = [(log / str(i)).read_text() for i in range(1, len(list(log.iterdir())) + 1)]
        index = (work / "results/eval_index.tsv").read_text().splitlines()
        return work, calls, index

    def tearDown(self):
        self.tmp.cleanup()

    def test_seven_jobs_with_template(self):
        work, calls, index = self.run_script("results/qwen4B_LoRA/p10-1/merged", "p10-ptst", "templates/ptst.jinja")
        self.assertEqual(len(calls), 7)
        self.assertIn("scripts/viking/agentharm.sbatch", calls[0])
        for bench in ("agentharm-harmful", "asb", "agent-safetybench", "bfcl-multiturn", "tau-bench"):
            self.assertTrue(any(f"agent_benchmarks.sbatch {bench}" in c for c in calls), bench)
        self.assertTrue(any("--time=24:00:00 scripts/viking/agent_benchmarks.sbatch bfcl-multiturn" in c for c in calls))
        self.assertEqual(sum("--time=" in c for c in calls), 1)
        harmbench = calls[6].split("\n")[0]
        model = str((work / "results/qwen4B_LoRA/p10-1/merged").resolve())
        self.assertIn(f"--model {model}", harmbench)
        self.assertIn(f"--chat-template {work.resolve() / 'templates/ptst.jinja'}", harmbench)
        for call in calls:
            self.assertIn(f"MODEL_PATH={model}", call)
            self.assertIn("MODEL_REVISION=<unset>", call)
            self.assertIn("--job-name=eval-p10-ptst-", call)
            self.assertIn("ptst.jinja", call.split("CHAT_TEMPLATE=")[1])
        self.assertEqual(index[0].split("\t"), ["tag", "benchmark", "jobid", "model_path", "chat_template_sha256"])
        self.assertEqual(len(index), 8)
        self.assertEqual(index[1].split("\t")[4], hashlib.sha256(b"T").hexdigest())

    def test_mosaic_scaffold(self):
        work, calls, index = self.run_script("results/qwen4B_LoRA/p10-1/merged", "p10", extra_env={"SCAFFOLD": "mosaic"})
        self.assertEqual(len(calls), 7)
        run = (work / "results/qwen4B_LoRA/p10-1").resolve()
        for call in calls:
            self.assertIn("--job-name=eval-p10-mosaic-", call)
            self.assertIn(f"CHAT_TEMPLATE={work.resolve() / 'templates/mosaic.jinja'}", call)
            self.assertIn("TOOL_CALL_PARSER=mosaic", call)
            self.assertIn("MOSAIC_SCAFFOLD=1", call)
            self.assertIn(f"RESULTS_DIR={run}/eval/mosaic", call)
        self.assertIn("--chat-template", calls[6])
        self.assertTrue(all(row.split("\t")[0] == "p10-mosaic" for row in index[1:]))
        sha = hashlib.sha256((ROOT / "templates/mosaic.jinja").read_bytes()).hexdigest()
        self.assertEqual(index[1].split("\t")[4], sha)

    def test_mosaic_scaffold_refuses_a_second_template(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_script("results/qwen4B_LoRA/p10-1/merged", "p10", "templates/ptst.jinja",
                            extra_env={"SCAFFOLD": "mosaic"})

    def test_hub_model_without_template(self):
        _, calls, index = self.run_script("Qwen/Qwen3-4B-Instruct-2507", "base", revision=None)
        self.assertTrue(all("MODEL_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554" in c for c in calls))
        self.assertTrue(all("CHAT_TEMPLATE=<unset>" in c for c in calls))
        self.assertNotIn("--chat-template", calls[6])
        self.assertEqual(index[1].split("\t")[4], "-")

    def test_qwen25_hub_model_pinned(self):
        _, calls, _ = self.run_script("Qwen/Qwen2.5-3B-Instruct", "qwen2.5-3B-base", revision=None)
        self.assertTrue(all("MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1" in c for c in calls))

    def test_glm_and_llama_hub_models_get_pin_and_training_template(self):
        for model, revision, template in (
                ("zai-org/glm-4-9b-chat-hf", "8599336fc6c125203efb2360bfaf4c80eef1d1bf", "glm4_tools.jinja"),
                ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "llama31_tools.jinja")):
            work, calls, index = self.run_script(model, "base", revision=None)
            path = work.resolve() / "templates" / template
            self.assertTrue(all(f"MODEL_REVISION={revision}" in c and f"CHAT_TEMPLATE={path}" in c for c in calls))
            self.assertIn(f"--chat-template {path}", calls[6])
            self.assertEqual(index[1].split("\t")[4], hashlib.sha256(path.read_bytes()).hexdigest())
            self.tmp.cleanup()

    def test_dangling_scratch_link_is_reported(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = self.tmp.name
        run = Path(tmp) / "results/glm4-9B_LoRA/p00-1"
        run.mkdir(parents=True)
        (run / "merged").symlink_to(Path(tmp) / "gone/merged")
        shutil.copytree(ROOT / "scripts", Path(tmp) / "scripts")
        result = subprocess.run(["bash", "scripts/viking/eval_all.sh", "results/glm4-9B_LoRA/p00-1/merged", "t"],
                                cwd=tmp, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no longer exists (scratch cleared?)", result.stderr)
        result = subprocess.run(["bash", "scripts/viking/lora_models.sh", "glm4-9B", "eval", "results/glm4-9B_LoRA/p00-1"],
                                cwd=tmp, capture_output=True, text=True)
        self.assertIn("no longer exists (scratch cleared?)", result.stderr)

    def test_unknown_hub_model_needs_revision(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_script("org/other-model", "other", revision=None)


class Qwen25DriverTests(unittest.TestCase):
    """scripts/viking/qwen25_3b.sh: training jobs carry the Qwen2.5 model, pin and tag."""

    def test_train_submits_tagged_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shutil.copytree(ROOT / "scripts", work / "scripts")
            (work / "trainset/data").mkdir(parents=True)
            for condition in ("p00", "p10"):
                (work / f"trainset/data/train_{condition}.jsonl").write_text("")
            bin_dir, log = work / "bin", work / "log"
            bin_dir.mkdir(), log.mkdir()
            (bin_dir / "sbatch").write_text(FAKE_SBATCH.replace("MODEL_REVISION=%s", "MODEL_REVISION=%s\\nTAG=%s COND=%s MAXLEN=%s")
                                            .replace('"${MODEL_REVISION:-<unset>}"',
                                                     '"${MODEL_REVISION:-<unset>}" "$MODEL_TAG" "$CONDITION" "$MAX_MODEL_LEN"'))
            (bin_dir / "sbatch").chmod(0o755)
            env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "LOG": str(log)}
            out = subprocess.run(["bash", "scripts/viking/qwen25_3b.sh", "train", "p00", "p10", "--", "--limit", "8"],
                                 cwd=work, env=env, check=True, capture_output=True, text=True).stdout
            calls = [(log / str(i)).read_text() for i in (1, 2)]
            for call, condition in zip(calls, ("p00", "p10")):
                self.assertIn(f"--job-name=lora-qwen2.5-3B-{condition} scripts/viking/train_lora.sbatch --merge --limit 8", call)
                self.assertIn("MODEL_PATH=Qwen/Qwen2.5-3B-Instruct", call)
                self.assertIn("MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1", call)
                self.assertIn(f"TAG=qwen2.5-3B COND={condition} MAXLEN=32768", call)
            self.assertIn("results/qwen2.5-3B_LoRA/p10-102", out)


class LoraModelsDriverTests(unittest.TestCase):
    """scripts/viking/lora_models.sh: GLM-4 / Llama-3.1 training jobs carry the model, pin, tag and a longer time limit."""

    def submit(self, model, *args, command="train", setup=None):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shutil.copytree(ROOT / "scripts", work / "scripts")
            shutil.copytree(ROOT / "templates", work / "templates")
            if setup:
                setup(work)
            (work / "trainset/data").mkdir(parents=True)
            for condition in ("p00", "p05", "p10", "p20"):
                (work / f"trainset/data/train_{condition}.jsonl").write_text("")
            bin_dir, log = work / "bin", work / "log"
            bin_dir.mkdir(), log.mkdir()
            (bin_dir / "sbatch").write_text(FAKE_SBATCH.replace("MODEL_REVISION=%s", "MODEL_REVISION=%s\\nTAG=%s COND=%s")
                                            .replace('"${MODEL_REVISION:-<unset>}"', '"${MODEL_REVISION:-<unset>}" "$MODEL_TAG" "$CONDITION"'))
            (bin_dir / "sbatch").chmod(0o755)
            env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "LOG": str(log)}
            out = subprocess.run(["bash", "scripts/viking/lora_models.sh", model, command, *args],
                                 cwd=work, env=env, check=True, capture_output=True, text=True).stdout
            return out, [(log / name).read_text() for name in sorted(os.listdir(log), key=int)]

    def test_glm_jobs(self):
        out, calls = self.submit("glm4-9B", "p00", "p10", "--", "--limit", "8")
        self.assertEqual(len(calls), 2)
        for call, condition in zip(calls, ("p00", "p10")):
            self.assertIn(f"--job-name=lora-glm4-9B-{condition} --time=24:00:00 scripts/viking/train_lora.sbatch --merge --limit 8", call)
            self.assertIn("MODEL_PATH=zai-org/glm-4-9b-chat-hf", call)
            self.assertIn("MODEL_REVISION=8599336fc6c125203efb2360bfaf4c80eef1d1bf", call)
            self.assertIn(f"TAG=glm4-9B COND={condition}", call)
        self.assertIn("results/glm4-9B_LoRA/p10-102", out)

    def test_llama_defaults_to_all_conditions(self):
        out, calls = self.submit("llama3.1-8B")
        self.assertEqual([c.split("COND=")[1].split()[0] for c in calls], ["p00", "p05", "p10", "p20"])
        self.assertIn("MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct", calls[0])
        self.assertIn("MODEL_REVISION=0e9e39f249a16976918f6564b8830bc894c89659", calls[0])
        self.assertIn("results/llama3.1-8B_LoRA/p20-104", out)

    def test_eval_base_serves_hub_model_with_training_template(self):
        _, calls = self.submit("glm4-9B", command="eval-base")
        self.assertEqual(len(calls), 7)
        for call in calls:
            self.assertIn("MODEL_PATH=zai-org/glm-4-9b-chat-hf", call)
            self.assertIn("MODEL_REVISION=8599336fc6c125203efb2360bfaf4c80eef1d1bf", call)
            self.assertIn("templates/glm4_tools.jinja", call.split("CHAT_TEMPLATE=")[1])
            self.assertIn("--job-name=eval-glm4-9B-base-", call)

    def test_eval_run_uses_merged_model(self):
        def make_run(work):
            merged = work / "results/llama3.1-8B_LoRA/p10-7/merged"
            merged.mkdir(parents=True)
            (merged / "config.json").write_text("{}")
            (merged / "chat_template.jinja").write_text("T")
        _, calls = self.submit("llama3.1-8B", "results/llama3.1-8B_LoRA/p10-7/", command="eval", setup=make_run)
        self.assertEqual(len(calls), 7)
        for call in calls:
            self.assertIn("results/llama3.1-8B_LoRA/p10-7/merged", call.split("MODEL_PATH=")[1].split("\n")[0])
            self.assertIn("MODEL_REVISION=<unset>", call)
            self.assertIn("CHAT_TEMPLATE=<unset>", call)
            self.assertIn("--job-name=eval-llama3.1-8B-p10-7-", call)

    def test_eval_refuses_run_without_saved_template(self):
        def make_run(work):
            merged = work / "results/glm4-9B_LoRA/p00-1/merged"
            merged.mkdir(parents=True)
            (merged / "config.json").write_text("{}")
        with self.assertRaises(subprocess.CalledProcessError):
            self.submit("glm4-9B", "results/glm4-9B_LoRA/p00-1", command="eval", setup=make_run)

    def test_unknown_model_is_refused(self):
        result = subprocess.run(["bash", "scripts/viking/lora_models.sh", "mistral", "train"], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


class ServeToolParserTests(unittest.TestCase):
    """scripts/eval/serve_qwen.sh picks the tool parser (and plugin) from the model's chat format."""

    def serve(self, model, **env):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            (bin_dir / "vllm").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
            (bin_dir / "vllm").chmod(0o755)
            if model.startswith("local:"):
                directory = Path(tmp) / "merged"
                directory.mkdir()
                (directory / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": {"1": {"content": model[6:]}}}))
                model = str(directory)
            out = subprocess.run(["bash", str(ROOT / "scripts/eval/serve_qwen.sh")], capture_output=True, text=True, check=True,
                                 env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "MODEL_PATH": model, **env}).stdout
            args = out.split("\n")
            parser = args[args.index("--tool-call-parser") + 1]
            plugin = args[args.index("--tool-parser-plugin") + 1] if "--tool-parser-plugin" in args else None
            return parser, plugin

    def test_parser_by_format(self):
        plugin = str(ROOT / "scripts/eval/tool_parsers.py")
        self.assertEqual(self.serve("Qwen/Qwen3-4B-Instruct-2507"), ("hermes", None))
        self.assertEqual(self.serve("local:<|im_start|>"), ("hermes", None))
        self.assertEqual(self.serve("zai-org/glm-4-9b-chat-hf"), ("glm4_native", plugin))
        self.assertEqual(self.serve("local:<|observation|>"), ("glm4_native", plugin))
        self.assertEqual(self.serve("meta-llama/Llama-3.1-8B-Instruct"), ("llama3_json_full", plugin))
        self.assertEqual(self.serve("local:<|start_header_id|>"), ("llama3_json_full", plugin))
        self.assertEqual(self.serve("local:<|observation|>", TOOL_CALL_PARSER="hermes"), ("hermes", None))


class ResultsDirTests(unittest.TestCase):
    """scripts/viking/results_dir.sh: results/<model>_<method>/<run> naming."""

    def call(self, expr, **env):
        script = f"source scripts/viking/results_dir.sh && {expr}"
        env = {**{k: v for k, v in os.environ.items() if k not in ("MODEL_TAG", "MODEL_PATH", "RESULTS_DIR")}, **env}
        return subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True, capture_output=True, text=True).stdout.strip()

    def test_training_groups(self):
        self.assertEqual(self.call("results_group lora --merge"), "results/qwen4B_LoRA")
        self.assertEqual(self.call("results_group lora", MODEL_TAG="qwen2.5-3B"), "results/qwen2.5-3B_LoRA")
        self.assertEqual(self.call("results_group lora Qwen/Qwen2.5-3B-Instruct"), "results/qwen2.5-3B_LoRA")
        self.assertEqual(self.call("results_group lora zai-org/glm-4-9b-chat-hf"), "results/glm4-9B_LoRA")
        self.assertEqual(self.call("results_group lora meta-llama/Llama-3.1-8B-Instruct"), "results/llama3.1-8B_LoRA")
        self.assertEqual(self.call("results_group safelora --adapter results/qwen2.5-3B_LoRA/p10-1/adapter"),
                         "results/qwen2.5-3B_SafeLoRA")
        self.assertEqual(self.call("results_group dpo /abs/results/qwen4B_LoRA/p10-1/merged"), "results/qwen4B_DPO")

    def test_eval_roots(self):
        self.assertEqual(self.call("eval_root"), "results/qwen4B_base")
        self.assertEqual(self.call("eval_root", MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"), "results/qwen2.5-3B_base")
        with tempfile.TemporaryDirectory() as tmp:
            merged = Path(tmp) / "results/qwen4B_LoRA/p10-1/merged"
            merged.mkdir(parents=True)
            self.assertEqual(self.call("eval_root", MODEL_PATH=str(merged)), str(merged.parent / "eval"))
            self.assertEqual(self.call(f"eval_root {merged}/"), str(merged.parent / "eval"))
        self.assertEqual(self.call("eval_root", RESULTS_DIR="results/x"), "results/x")


if __name__ == "__main__":
    unittest.main()
