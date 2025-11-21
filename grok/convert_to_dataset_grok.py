import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

# Ensure repository root is on sys.path so imports like `gpt.*` and `prompts.*`
# work when running this script from the `grok/` folder directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from prompts.utils import get_task_name

# Reuse task converters from the OpenAI converter to keep behavior consistent
from gpt.convert_to_dataset import TASK_CONVERTERS, pairs_to_examples


def extract_output_text_grok(entry: Dict[str, Any]) -> str | None:
    # Try common shapes produced by grok_generate.py and by Grok API responses
    # 1) OpenAI-shaped: entry['response']['body']['output'][0]['content'][0]['text']
    response = entry.get("response")
    if isinstance(response, dict):
        body = response.get("body")
        if isinstance(body, dict):
            output_list = body.get("output")
            if output_list and isinstance(output_list, list):
                message = output_list[0]
                content_list = message.get("content", []) if isinstance(message, dict) else []
                for content in content_list:
                    if not isinstance(content, dict):
                        continue
                    if content.get("type") in ("output_text", "text"):
                        text = content.get("text")
                        if isinstance(text, str):
                            text = text.strip()
                            return text or None

    # 2) Grok/Chat style: entry may directly include choices -> message -> content/text
    if isinstance(entry, dict):
        # check top-level choices
        choices = entry.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message") or first
                if isinstance(msg, dict):
                    # message.content may be string or dict
                    content = msg.get("content") or msg.get("text")
                    if isinstance(content, str):
                        return content.strip() or None
                    if isinstance(content, list) or isinstance(content, dict):
                        try:
                            return json.dumps(content, ensure_ascii=False)
                        except Exception:
                            return str(content)

    # 3) fallback: sometimes the entry itself is a plain string in an envelope
    # try common fields
    for key in ("text", "output", "body"):
        val = entry.get(key)
        if isinstance(val, str):
            return val.strip() or None

    return None


def load_pairs_from_batch_output_grok(path: Path, task: str) -> List[Dict[str, Any]]:
    all_examples: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] {path.name} Line {line_idx}: invalid JSON, skipping.")
                continue

            text = extract_output_text_grok(entry)
            if not text:
                print(f"[WARN] {path.name} Line {line_idx}: no output text found, skipping.")
                continue

            # The generated text should be JSON that contains 'pairs' per our prompt schema.
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                print(f"[WARN] {path.name} Line {line_idx}: output text is not valid JSON, skipping.")
                continue

            pairs = parsed.get("pairs")
            if not isinstance(pairs, list):
                print(f"[WARN] {path.name} Line {line_idx}: 'pairs' not found or not a list, skipping.")
                continue

            examples = pairs_to_examples(task, pairs)
            all_examples.extend(examples)

    return all_examples


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig):
    task = get_task_name(cfg)
    model_name = cfg.model

    base_dir_rel = Path("data") / task / model_name
    base_dir = Path(to_absolute_path(str(base_dir_rel)))

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Input dir not found: {base_dir}\n"
            f"Expected batch outputs at data/{task}/{model_name}/batchoutput_*.jsonl"
        )

    response_paths: List[Path] = sorted(base_dir.glob("batchoutput_*.jsonl"))
    if not response_paths:
        raise FileNotFoundError(
            f"No batchoutput_*.jsonl files found in {base_dir}.\n"
            f"Check that retrieve script saved files as batchoutput_XXXX.jsonl."
        )

    output_path = base_dir / "synthetic_grok.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Task        : {task}")
    print(f"[INFO] Model       : {model_name}")
    print(f"[INFO] Input dir   : {base_dir}")
    print(f"[INFO] Num inputs  : {len(response_paths)}")
    print(f"[INFO] Input files :")
    for p in response_paths:
        print(f"  - {p.name}")
    print(f"[INFO] Output file : {output_path}")

    all_examples: List[Dict[str, Any]] = []
    for path in response_paths:
        examples = load_pairs_from_batch_output_grok(path, task)
        print(f"[INFO] {path.name}: collected {len(examples)} examples.")
        all_examples.extend(examples)

    print(f"[INFO] Total collected {len(all_examples)} examples. Writing JSONL...")

    with output_path.open("w", encoding="utf-8") as out_f:
        for ex in all_examples:
            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
