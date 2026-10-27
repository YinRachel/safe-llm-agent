"""scripts/viking_eval_all.sh against a fake sbatch: seven jobs, unchanged scripts, model routing."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
FAKE_SBATCH = """#!/usr/bin/env bash
n=$(( $(ls "$LOG" | wc -l) + 1 ))
{ printf 'ARGS'; printf ' %s' "$@"; printf '\\nMODEL_PATH=%s\\nMODEL_REVISION=%s\\n' \\
  "${MODEL_PATH:-}" "${MODEL_REVISION:-<unset>}"; } > "$LOG/$n"
echo "10$n"
"""


class EvalAllTests(unittest.TestCase):
    def run_script(self, *args, revision="should-be-unset"):
        self.tmp = tempfile.TemporaryDirectory()
        work = Path(self.tmp.name)
        shutil.copytree(ROOT / "scripts", work / "scripts")
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
        subprocess.run(["bash", "scripts/viking_eval_all.sh", *args], cwd=work, env=env, check=True, capture_output=True)
        calls = [(log / str(i)).read_text() for i in range(1, len(list(log.iterdir())) + 1)]
        index = (work / "results/eval_index.tsv").read_text().splitlines()
        return work, calls, index

    def tearDown(self):
        self.tmp.cleanup()

    def test_seven_jobs_for_local_model(self):
        work, calls, index = self.run_script("results/lora-p10-1/merged", "p10")
        self.assertEqual(len(calls), 7)
        self.assertIn("scripts/viking_agentharm.sbatch", calls[0])
        for bench in ("agentharm-harmful", "asb", "agent-safetybench", "bfcl-multiturn", "tau-bench"):
            self.assertTrue(any(f"viking_agent_benchmarks.sbatch {bench}" in c for c in calls), bench)
        model = str((work / "results/lora-p10-1/merged").resolve())
        self.assertIn(f"viking_harmbench.sbatch --model {model}", calls[6].split("\n")[0])
        for call in calls:
            self.assertIn(f"MODEL_PATH={model}", call)
            self.assertIn("MODEL_REVISION=<unset>", call)
            self.assertIn("--job-name=eval-p10-", call)
        self.assertEqual(index[0].split("\t"), ["tag", "benchmark", "jobid", "model_path"])
        self.assertEqual(len(index), 8)
        self.assertEqual(index[1].split("\t"), ["p10", "agentharm_benign", "101", model])
        self.assertTrue((work / "logs").is_dir())

    def test_hub_model_gets_pinned_revision(self):
        _, calls, index = self.run_script("Qwen/Qwen3-4B-Instruct-2507", "base", revision=None)
        self.assertTrue(all("MODEL_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554" in c for c in calls))
        self.assertTrue(all("MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507" in c for c in calls))
        self.assertEqual(len(index), 8)


if __name__ == "__main__":
    unittest.main()
