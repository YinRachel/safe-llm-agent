"""F4 PTST ("Pure Tuning, Safe Testing"): a serving-time chat template with a safety system prompt.

Lyu et al., *Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates*, NeurIPS 2024,
arXiv:2402.18540 (https://github.com/vfleaking/PTST @ 134600f): fine-tune without a safety prompt, test with one.

The safety text is AgentHarm's own published refusal instruction, imported from inspect_evals
(`inspect_evals.agentharm.prompts.prompt_technique_dict["refusal"]`, surrounding whitespace stripped). The checkpoint's
chat template (Qwen3-4B-Instruct-2507 at the pinned revision, read from the local HF cache) is kept verbatim; a Jinja
preamble rewrites `messages` first:
  * a leading system message gets "\\n\\n" + SAFETY_TEXT appended to its content;
  * otherwise a system message with SAFETY_TEXT is inserted.
Qwen's template then builds the system block itself (with tools when present), so the safety text lands inside the
system block in both branches and the rendering is otherwise byte-identical (tests/test_safety_template.py).

  .venv/bin/python scripts/post-training/safety_template.py --out templates/ptst.jinja     # writes ptst.jinja + ptst.json
Serve with CHAT_TEMPLATE=templates/ptst.jinja (scripts/eval/serve_qwen.sh, generate_harmbench.py --chat-template).
"""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PREAMBLE_MARK = "{#- PTST safety preamble (scripts/post-training/safety_template.py) #}"


def load_safety_text():
    from inspect_evals.agentharm.prompts import prompt_technique_dict
    return prompt_technique_dict["refusal"].strip()


def load_chat_template(model=MODEL, revision=REVISION):
    """The checkpoint's own template from tokenizer_config.json in the local HF cache (no download)."""
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(model, "tokenizer_config.json", revision=revision)
    if not isinstance(path, str):
        raise SystemExit(f"tokenizer_config.json of {model}@{revision} is not in the local HF cache; download it first "
                         "(scripts/viking/download.sh on the login node).")
    return json.loads(Path(path).read_text())["chat_template"]


def make_template(original, safety_text):
    for bad in ('"', "\\", "{", "}", "%"):
        if bad in safety_text:
            raise ValueError(f"safety text contains {bad!r}; it is embedded as a Jinja string literal")
    preamble = (
        PREAMBLE_MARK
        + f'{{%- set ptst_safety = "{safety_text}" %}}'
        + "{%- if messages and messages[0].role == 'system' %}"
        + "{%- set messages = [{'role': 'system', 'content': messages[0].content + '\\n\\n' + ptst_safety}] + messages[1:] %}"
        + "{%- else %}"
        + "{%- set messages = [{'role': 'system', 'content': ptst_safety}] + messages %}"
        + "{%- endif %}"
    )
    return preamble + original


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="templates/ptst.jinja")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args(argv)
    original, safety = load_chat_template(args.model, args.revision), load_safety_text()
    template = make_template(original, safety)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template)
    from importlib.metadata import version
    meta = {"model": args.model, "revision": args.revision, "original_template_sha256": sha256_text(original),
            "template_sha256": sha256_text(template), "safety_text": safety, "safety_text_sha256": sha256_text(safety),
            "safety_source": "inspect_evals.agentharm.prompts.prompt_technique_dict['refusal'] (stripped)",
            "inspect_evals_version": version("inspect_evals"),
            "method": "PTST (Lyu et al., NeurIPS 2024): appended to the system message, or inserted as one"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} (sha256 {meta['template_sha256'][:16]}) and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
