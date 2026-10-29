"""scripts/viking/eval_all.sh against a fake sbatch: seven jobs, unchanged scripts, template and model routing."""
import hashlib
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
  "${MODEL_PATH:-}" "${MODEL_REVISION:-<unset>}" "${CHAT_TEMPLATE:-<unset>}"; } > "$LOG/$n"
echo "10$n"
"""


class EvalAllTests(unittest.TestCase):
    def run_script(self, *args, revision="should-be-unset"):
        self.tmp = tempfile.TemporaryDirectory()
        work = Path(self.tmp.name)
        shutil.copytree(ROOT / "scripts", work / "scripts")
        (work / "templates").mkdir()
        (work / "templates/ptst.jinja").write_text("T")
        model = work / "results/qwen4B_LoRA/p10-1/merged"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        bin_dir, log = work / "bin", work / "log"
        bin_dir.mkdir(), log.mkdir()
        (bin_dir / "sbatch").write_text(FAKE_SBATCH)
        (bin_dir / "sbatch").chmod(0o755)
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "LOG": str(log)}
        env.pop("MODEL_REVISION", None)
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

    def test_hub_model_without_template(self):
        _, calls, index = self.run_script("Qwen/Qwen3-4B-Instruct-2507", "base", revision=None)
        self.assertTrue(all("MODEL_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554" in c for c in calls))
        self.assertTrue(all("CHAT_TEMPLATE=<unset>" in c for c in calls))
        self.assertNotIn("--chat-template", calls[6])
        self.assertEqual(index[1].split("\t")[4], "-")

    def test_qwen25_hub_model_pinned(self):
        _, calls, _ = self.run_script("Qwen/Qwen2.5-3B-Instruct", "qwen2.5-3B-base", revision=None)
        self.assertTrue(all("MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1" in c for c in calls))

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
