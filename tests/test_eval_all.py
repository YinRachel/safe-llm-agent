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
        model = work / "results/lora-p10-1/merged"
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
        work, calls, index = self.run_script("results/lora-p10-1/merged", "p10-ptst", "templates/ptst.jinja")
        self.assertEqual(len(calls), 7)
        self.assertIn("scripts/viking/agentharm.sbatch", calls[0])
        for bench in ("agentharm-harmful", "asb", "agent-safetybench", "bfcl-multiturn", "tau-bench"):
            self.assertTrue(any(f"agent_benchmarks.sbatch {bench}" in c for c in calls), bench)
        harmbench = calls[6].split("\n")[0]
        model = str((work / "results/lora-p10-1/merged").resolve())
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


if __name__ == "__main__":
    unittest.main()
