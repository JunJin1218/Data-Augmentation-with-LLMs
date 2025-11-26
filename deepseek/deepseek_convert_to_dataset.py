import json
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from prompts.utils import get_task_name


# =========================
# Helper: extract text from batchoutput entry
# =========================

def extract_output_text(entry: Dict[str, Any]) -> Optional[str]:
    """
    Extract the model's text from a batchoutput JSON line.

    We expect shape like:

      {
        "response": {
          "body": {
            "output": [
              {
                "content": [
                  {
                    "type": "output_text",
                    "text": "<JSON string or ```json fenced block>"
                  }
                ]
              }
            ]
          }
        }
      }
    """
    response = entry.get("response")
    if not isinstance(response, dict):
        return None

    body = response.get("body")
    if not isinstance(body, dict):
        return None

    outputs = body.get("output")
    if not isinstance(outputs, list) or not outputs:
        return None

    first_output = outputs[0]
    if not isinstance(first_output, dict):
        return None

    contents = first_output.get("content")
    if not isinstance(contents, list) or not contents:
        return None

    for c in contents:
        if not isinstance(c, dict):
            continue
        text = c.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    return None


def strip_markdown_fence(text: str) -> str:
    """
    Remove ```...``` Markdown fences around a JSON block if present.

    Handles things like:
      ```json
      { ... }
      ```
    or
      ```
      { ... }
      ```
    """
    s = text.strip()
    if not s.startswith("```"):
        return s

    # Drop the first line (``` or ```json)
    first_newline = s.find("\n")
    if first_newline != -1:
        s = s[first_newline + 1 :]
    else:
        # weird single-line ```... case; just return as-is
        return text

    # Remove trailing ``` if present
    fence_pos = s.rfind("```")
    if fence_pos != -1:
        s = s[:fence_pos]

    return s.strip()


# =========================
# Task-specific converters
# =========================

def convert_glue_mrpc(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert schema.json 'pairs' to GLUE MRPC-style examples.
    """
    examples: List[Dict[str, Any]] = []

    for p in pairs:
        if not isinstance(p, dict):
            continue

        s1 = (
            p.get("Sentence1")
            or p.get("sentence1")
            or p.get("sent1")
        )
        s2 = (
            p.get("Sentence2")
            or p.get("sentence2")
            or p.get("sent2")
        )
        lbl = p.get("Label")
        if lbl is None:
            lbl = p.get("label")

        if not isinstance(s1, str) or not isinstance(s2, str):
            continue

        try:
            label_int = int(lbl) if lbl is not None else None
        except (TypeError, ValueError):
            label_int = None

        ex = {"sentence1": s1, "sentence2": s2}
        if label_int is not None:
            ex["label"] = label_int

        examples.append(ex)

    return examples


def convert_super_glue_cb(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert schema.json 'pairs' to SuperGLUE CB-style examples.

    Input items:

      {"Premise": "...", "Hypothesis": "...", "Label": 0|1|2}

    Output items:

      {"premise": "...", "hypothesis": "...", "label": 0|1|2}
    """
    examples: List[Dict[str, Any]] = []

    for p in pairs:
        if not isinstance(p, dict):
            continue

        premise = p.get("Premise") or p.get("premise")
        hypothesis = p.get("Hypothesis") or p.get("hypothesis")
        lbl = p.get("Label")
        if lbl is None:
            lbl = p.get("label")

        if not isinstance(premise, str) or not isinstance(hypothesis, str):
            continue

        try:
            label_int = int(lbl) if lbl is not None else None
        except (TypeError, ValueError):
            label_int = None

        ex = {"premise": premise, "hypothesis": hypothesis}
        if label_int is not None:
            ex["label"] = label_int

        examples.append(ex)

    return examples


TASK_CONVERTERS: Dict[str, Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = {
    "glue-mrpc": convert_glue_mrpc,
    "super_glue-cb": convert_super_glue_cb,
}


def pairs_to_examples(task: str, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if task not in TASK_CONVERTERS:
        raise ValueError(
            f"No converter registered for task '{task}'. "
            f"Known tasks: {list(TASK_CONVERTERS.keys())}"
        )
    return TASK_CONVERTERS[task](pairs)


# =========================
# IO: load batchoutput_*.jsonl → list of examples
# =========================

def load_pairs_from_batch_output(path: Path, task: str) -> List[Dict[str, Any]]:
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

            text = extract_output_text(entry)
            if not text:
                print(f"[WARN] {path.name} Line {line_idx}: no output text found, skipping.")
                continue

            # Remove ``` fences if DeepSeek wrapped the JSON in a markdown block
            json_str = strip_markdown_fence(text)

            # Try to parse the entire text as a single JSON object first.
            parsed = None
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # If that fails, the model may have emitted NDJSON (multiple JSON objects
                # separated by newlines). Try parsing line-by-line and collecting any
                # objects we can decode.
                ndjson_objs: List[Dict[str, Any]] = []
                for subline in json_str.splitlines():
                    subline = subline.strip()
                    if not subline:
                        continue
                    try:
                        obj = json.loads(subline)
                        if isinstance(obj, dict):
                            ndjson_objs.append(obj)
                    except json.JSONDecodeError:
                        # ignore non-JSON lines
                        continue

                # If we found NDJSON objects, use them directly. They may be either
                # (a) a sequence of pair objects (each with 'premise'/'hypothesis'), or
                # (b) JSON objects containing a top-level 'pairs' list. Detect both.
                if ndjson_objs:
                    # If any ndjson object contains a 'pairs' list, prefer that object's pairs.
                    found_pairs = None
                    for o in ndjson_objs:
                        if isinstance(o.get("pairs"), list):
                            found_pairs = o.get("pairs")
                            break

                    if found_pairs is not None:
                        pairs = found_pairs
                    else:
                        # Otherwise, assume the NDJSON objects themselves are pair dicts.
                        pairs = ndjson_objs

                    examples = pairs_to_examples(task, pairs)
                    all_examples.extend(examples)
                    continue

            # If parsed succeeded as a dict or list, normalize different shapes:
            if isinstance(parsed, dict):
                # Accept either a top-level 'pairs' list or a single pair dict.
                pairs = parsed.get("pairs")
                if isinstance(pairs, list):
                    pass
                elif all(k in parsed for k in ("premise", "hypothesis")):
                    pairs = [parsed]
                else:
                    print(
                        f"[WARN] {path.name} Line {line_idx}: 'pairs' not found or invalid, skipping."
                    )
                    continue
            elif isinstance(parsed, list):
                # Parsed as a JSON list of pair objects
                pairs = parsed
            else:
                print(
                    f"[WARN] {path.name} Line {line_idx}: output text is not valid JSON or NDJSON, skipping."
                )
                continue

            examples = pairs_to_examples(task, pairs)
            all_examples.extend(examples)

    return all_examples


# =========================
# Main Hydra entry point
# =========================

@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig) -> None:
    """
    Convert DeepSeek batch outputs into a flat JSONL dataset.

    Expects:
      - task from get_task_name(cfg), e.g. "super_glue-cb"
      - model name in cfg.model (default: "deepseek-chat")
      - input: data/{task}/{model}/batchoutput_*.jsonl

    Writes:
      - data/{task}/{model}/synthetic.jsonl
    """
    task = get_task_name(cfg)
    model_name = cfg.get("model", "deepseek-chat")

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
            f"Check that deepseek_retreive.py saved files as batchoutput_XXXX.jsonl."
        )

    output_path = base_dir / "synthetic.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Task        : {task}")
    print(f"[INFO] Model       : {model_name}")
    print(f"[INFO] Input dir   : {base_dir}")
    print(f"[INFO] Input files :")
    for p in response_paths:
        print(f"  - {p.name}")
    print(f"[INFO] Output file : {output_path}")

    all_examples: List[Dict[str, Any]] = []
    for path in response_paths:
        examples = load_pairs_from_batch_output(path, task)
        print(f"[INFO] {path.name}: collected {len(examples)} examples.")
        all_examples.extend(examples)

    print(f"[INFO] Total collected {len(all_examples)} examples. Writing JSONL...")

    with output_path.open("w", encoding="utf-8") as out_f:
        for ex in all_examples:
            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
