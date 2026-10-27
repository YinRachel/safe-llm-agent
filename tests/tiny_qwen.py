"""Tiny offline Qwen3 stand-in for trainer tests: a character-level fast tokenizer with a Qwen-shaped chat template
(tools in the system block, assistant turns `<|im_start|>assistant ... <|im_end|>`) and a randomly initialised 2-layer
Qwen3. Nothing is downloaded. Conversations use placeholder strings only."""
import json
from pathlib import Path
import string

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

# No Jinja whitespace control: every newline below is output text.
CHAT_TEMPLATE = (
    "{% if tools %}<|im_start|>system\n{% if messages[0].role == 'system' %}{{ messages[0].content }}\n\n{% endif %}"
    "# Tools\n{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}<|im_end|>\n"
    "{% elif messages[0].role == 'system' %}<|im_start|>system\n{{ messages[0].content }}<|im_end|>\n{% endif %}"
    "{% for message in messages %}"
    "{% if message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end|>\n"
    "{% elif message.role == 'assistant' %}<|im_start|>assistant\n{{ message.content }}"
    "{% for call in message.tool_calls or [] %}<tool_call>\n{{ call.function | tojson }}\n</tool_call>{% endfor %}<|im_end|>\n"
    "{% elif message.role == 'tool' %}<|im_start|>user\n<tool_response>\n{{ message.content }}\n</tool_response><|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def make_tokenizer():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    chars = sorted(set(string.printable))
    vocab = {token: i for i, token in enumerate(SPECIAL + ["[UNK]"] + chars)}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", eos_token="<|im_end|>",
                                        pad_token="<|endoftext|>", additional_special_tokens=SPECIAL[1:])
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def make_model_dir(path, seed=0, layers=2):
    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config
    path = Path(path)
    tokenizer = make_tokenizer()
    config = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, vocab_size=len(tokenizer), max_position_embeddings=4096,
                         tie_word_embeddings=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path


TOOLS = [{"type": "function", "function": {"name": "tool_a", "description": "placeholder",
                                           "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]


def conversation(i, label="benign", turns=2):
    """A multi-turn tool-use conversation of variable length (so padding/normalisation matter)."""
    messages = [{"role": "system", "content": "SYSTEM_PLACEHOLDER"}, {"role": "user", "content": f"USER_PLACEHOLDER {i} " + "x" * (i % 7)}]
    for t in range(turns):
        call = {"id": f"c{t}", "type": "function", "function": {"name": "tool_a", "arguments": json.dumps({"q": f"{i}-{t}"})}}
        messages += [{"role": "assistant", "content": "", "tool_calls": [call]},
                     {"role": "tool", "content": "TOOL_OUTPUT_PLACEHOLDER " + "y" * ((i + t) % 5), "tool_call_id": f"c{t}"}]
    messages.append({"role": "assistant", "content": f"ANSWER_PLACEHOLDER {i} " + "z" * (i % 11)})
    return {"id": f"{label}-{i:03d}", "source": "synthetic", "label": label, "category": label, "sub_category": "s",
            "tools": TOOLS, "messages": messages, "n_tokens": 0}


def refusal(i):
    return {"id": f"refusal-{i:03d}", "source": "synthetic", "label": "harmful", "category": "c", "sub_category": "s",
            "tools": TOOLS, "messages": [{"role": "system", "content": "SYSTEM_PLACEHOLDER"},
                                         {"role": "user", "content": f"HARMFUL_PLACEHOLDER {i}"},
                                         {"role": "assistant", "content": "REFUSAL_PLACEHOLDER " + "r" * (i % 4)}],
            "n_tokens": 0}


def write_jsonl(path, rows):
    with open(path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return str(path)
