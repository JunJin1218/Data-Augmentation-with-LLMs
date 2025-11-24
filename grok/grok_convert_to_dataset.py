import json
import re
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from prompts.utils import get_task_name


def extract_output_text(entry: Dict[str, Any]) -> Optional[str]:
    """
    Extract text output from either:
      - Direct Grok output: {"grok_output": "<text>"}
      - OpenAI-style / wrapped batch output:
        entry["response"]["body"]["output"][0]["content"][*]["text"]
    """

    # 1) Direct Grok output (what grok_retreive writes)
    grok_text = entry.get("grok_output")
    if isinstance(grok_text, str):
        grok_text = grok_text.strip()
        if grok_text:
            return grok_text

    # 2) OpenAI-style / batch output (if you ever wrap Grok like OpenAI)
    response = entry.get("response")
    if response is None:
        return None

    body = response.get("body")
    if body is None:
        return None

    output_list = body.get("output")
    if not output_list:
        return None

    message = output_list[0]
    content_list = message.get("content", [])
    if not content_list:
        return None

    for content in content_list:
        if content.get("type") in ("output_text", "text"):
            text = content.get("text")
            if isinstance(text, str):
                text = text.strip()
                return text or None

    return None


# -------- text → list of {"Premise","Hypothesis","Label"} --------

def extract_pairs_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse Grok's output into a list of dicts with keys:
      - "Premise"
      - "Hypothesis"
      - "Label"  (0/1/2, will be cast to int)

    Handles all of these formats:

      1) Pure JSON:
         {"pairs":[{...},{...}]}
         or
         [{...},{...}]

      2) Prose with embedded JSON objects:
         Example 1:
         {
           "Premise": "...",
           "Hypothesis": "...",
           "Label": 0  // comment
         }
         Example 2:
         { ... }

      3) Prose with labeled lines:
         Premise: ...
         Hypothesis: ...
         Label: 0
    """
    text = text.strip()
    if not text:
        return None

    # --- helper: try to parse a string as JSON and return a list of pairs ---
    def parse_candidate(s: str) -> Optional[List[Dict[str, Any]]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None

        # Case 1: {"pairs": [ ... ]}
        if isinstance(obj, dict) and "pairs" in obj:
            pairs = obj["pairs"]
            if isinstance(pairs, list):
                return pairs
            return None

        # Case 2: top-level list [ {...}, {...}, ... ]
        if isinstance(obj, list):
            return obj

        return None

    # --- 1) Direct JSON: whole text is JSON ---
    pairs = parse_candidate(text)
    if pairs is not None:
        return pairs

    # --- 2) Markdown fences ``` ... ``` ---
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json"):
                part = part[4:].strip()
            pairs = parse_candidate(part)
            if pairs is not None:
                return pairs

    # --- 3) Embedded JSON objects with Premise/Hypothesis/Label anywhere ---
    # Find each { ... } that mentions those keys in some order
    snippets = re.findall(
        r'\{[^{}]*"Premise"[^{}]*"Hypothesis"[^{}]*"Label"[^{}]*\}',
        text,
        flags=re.DOTALL,
    )

    parsed_pairs: List[Dict[str, Any]] = []
    for snip in snippets:
        # Remove '//' comments inside the object (Grok sometimes adds explanations)
        snippet_no_comments = re.sub(r'//.*', '', snip)
        try:
            obj = json.loads(snippet_no_comments)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        prem = obj.get("Premise")
        hyp = obj.get("Hypothesis")
        lab = obj.get("Label")
        if (
            isinstance(prem, str)
            and isinstance(hyp, str)
            and (isinstance(lab, int) or isinstance(lab, float))
        ):
            parsed_pairs.append(
                {"Premise": prem.strip(), "Hypothesis": hyp.strip(), "Label": int(lab)}
            )

    if parsed_pairs:
        return parsed_pairs

    # --- 4) "Premise: ... Hypothesis: ... Label: X" patterns anywhere ---
    line_pairs: List[Dict[str, Any]] = []
    # DOTALL lets Premise/Hypothesis span multiple lines
    pattern = re.compile(
        r'Premise:\s*(.+?)\s*Hypothesis:\s*(.+?)\s*Label:\s*([012])',
        flags=re.DOTALL | re.IGNORECASE,
    )

    for prem, hyp, lab in pattern.findall(text):
        prem = prem.replace("  \n", " ").strip()
        hyp = hyp.replace("  \n", " ").strip()
        line_pairs.append(
            {"Premise": prem, "Hypothesis": hyp, "Label": int(lab)}
        )

    if line_pairs:
        return line_pairs

    # --- 5) Last-resort: grab the largest {...} or [...] chunk and try again ---
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj : end_obj + 1]
        pairs = parse_candidate(candidate)
        if pairs is not None:
            return pairs

    start_list = text.find("[")
    end_list = text.rfind("]")
    if start_list != -1 and end_list != -1 and end_list > start_list:
        candidate = text[start_list : end_list + 1]
        pairs = parse_candidate(candidate)
        if pairs is not None:
            return pairs

    return None


# =========================
# Task-specific converters
# =========================

def convert_glue_mrpc(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert schema.json 'pairs' to GLUE MRPC-style examples.
    """
    examples: List[Dict[str, Any]] = []
    for pair in pairs:
        sentence1 = pair.get("Text1")
        sentence2 = pair.get("Text2")
        equivalence = pair.get("Equivalence")
        if sentence1 is None or sentence2 is None or equivalence is None:
            continue
        label = 1 if bool(equivalence) else 0
        examples.append(
            {
                "sentence1": sentence1,
                "sentence2": sentence2,
                "label": label,
            }
        )
    return examples


def convert_super_glue_cb(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert CB-style 'pairs' to SuperGLUE CB examples.

    Each pair is expected to have:
      - "Premise"
      - "Hypothesis"
      - "Label"
    """
    examples: List[Dict[str, Any]] = []
    for pair in pairs:
        premise = pair.get("Premise")
        hypothesis = pair.get("Hypothesis")
        label = pair.get("Label")
        if premise is None or hypothesis is None or label is None:
            continue
        examples.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": int(label),
            }
        )
    return examples


TASK_CONVERTERS: Dict[str, Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = {
    "glue-mrpc": convert_glue_mrpc,
    "super_glue-cb": convert_super_glue_cb,
}


def pairs_to_examples(task: str, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converter = TASK_CONVERTERS.get(task)
    if converter is None:
        raise ValueError(f"Unsupported task for conversion: {task}")
    return converter(pairs)


def load_pairs_from_batch_output(path: Path, task: str) -> List[Dict[str, Any]]:
    """
    Read a batchoutput_*.jsonl file and convert all parsed pairs
    into dataset-style examples.
    """
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

            pairs = extract_pairs_from_text(text)
            if not pairs:
                print(
                    f"[WARN] {path.name} Line {line_idx}: "
                    f"could not extract any CB-style pairs from text, skipping."
                )
                continue

            examples = pairs_to_examples(task, pairs)
            all_examples.extend(examples)

    return all_examples


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig):
    """
    Convert Grok batch outputs into a synthetic dataset.

    cfg example:
      - model: grok-3-mini
      - dataset: super_glue
      - subset: cb
      - batch: 1000
      - shots: 5

    I/O paths:
      - input:  data/{task_name}/{model_name}/batchoutput_*.jsonl
      - output: data/{task_name}/{model_name}/synthetic.jsonl
    """
    task = get_task_name(cfg)       # e.g. "super_glue-cb"
    model_name = cfg.model          # e.g. "grok-3-mini"

    base_dir_rel = Path("data") / task / model_name
    base_dir = Path(to_absolute_path(str(base_dir_rel)))

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Input dir not found: {base_dir}\n"
            f"Expected batch outputs at data/{task}/{model_name}/batchoutput_*.jsonl"
        )

    response_paths = sorted(base_dir.glob("batchoutput_*.jsonl"))

    if not response_paths:
        raise FileNotFoundError(
            f"No batchoutput_*.jsonl files found in {base_dir}.\n"
            f"Check that retrieve script saved files as batchoutput_XXXX.jsonl."
        )

    output_path = base_dir / "synthetic.jsonl"
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
