import json
from pathlib import Path
from typing import List, Dict, Any, Callable

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from prompts.utils import get_task_name


def extract_output_text(entry: Dict[str, Any]) -> str | None:
    response = entry.get("response")
    if response is None:
        return None

    body = response.get("body")
    if body is None:
        return None

    output_list = body.get("output")
    if not output_list:
        return None

    # 그냥 첫 message만 사용
    message = output_list[0]

    content_list = message.get("content", [])
    if not content_list:
        return None

    # 첫 번째 output_text (또는 text)만 사용
    for content in content_list:
        if content.get("type") in ("output_text", "text"):
            text = content.get("text")
            if isinstance(text, str):
                text = text.strip()
                return text or None

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
    Convert schema.json 'pairs' to SuperGLUE CB-style examples.
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


# 여기에 태스크별 컨버터를 계속 추가하면 됨
# def convert_super_glue_rte(...): ...
# def convert_sst2(...): ...


TASK_CONVERTERS: Dict[str, Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = {
    "glue-mrpc": convert_glue_mrpc,
    "super_glue-cb": convert_super_glue_cb,
    # "super_glue-rte": convert_super_glue_rte,
    # "glue-sst2": convert_glue_sst2,
}


def pairs_to_examples(task: str, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dispatch to a task-specific converter based on `task` name.
    """
    converter = TASK_CONVERTERS.get(task)
    if converter is None:
        raise ValueError(f"Unsupported task for conversion: {task}")
    return converter(pairs)


def load_pairs_from_batch_output(path: Path, task: str) -> List[Dict[str, Any]]:
    """
    Read an OpenAI batch output .jsonl file for /v1/responses
    and convert all 'pairs' into dataset-style examples.
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

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                print(
                    f"[WARN] {path.name} Line {line_idx}: output text is not valid JSON, skipping."
                )
                continue

            pairs = parsed.get("pairs")
            if not isinstance(pairs, list):
                print(
                    f"[WARN] {path.name} Line {line_idx}: 'pairs' not found or not a list, skipping."
                )
                continue

            examples = pairs_to_examples(task, pairs)
            all_examples.extend(examples)

    return all_examples


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig):
    """
    Convert OpenAI batch output (.jsonl) into a synthetic dataset (.jsonl).

    cfg에는 다음 필드만 있다고 가정:
      - model: gpt-4o-mini
      - dataset: super_glue
      - subset: cb
      - batch: 1000
      - shots: 5

    입출력 경로:
      - 입력: data/{task_name}/{model_name}/batchoutput_*.jsonl
      - 출력: data/{task_name}/{model_name}/synthetic.jsonl
    """
    task = get_task_name(cfg)       # 예: "super_glue-cb"
    model_name = cfg.model          # 예: "gpt-4o-mini"

    # -------------------------
    # 1) 입력 디렉토리 & 파일들
    # -------------------------
    # base_dir_rel = data/{task}/{model}
    base_dir_rel = Path("data") / task / model_name
    base_dir = Path(to_absolute_path(str(base_dir_rel)))

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Input dir not found: {base_dir}\n"
            f"Expected batch outputs at data/{task}/{model_name}/batchoutput_*.jsonl"
        )

    # 네가 retrieve에서 batchoutput_XXXX.jsonl 로 저장했으니까 그 패턴만 긁자
    response_paths: List[Path] = sorted(base_dir.glob("batchoutput_*.jsonl"))

    if not response_paths:
        raise FileNotFoundError(
            f"No batchoutput_*.jsonl files found in {base_dir}.\n"
            f"Check that retrieve script saved files as batchoutput_XXXX.jsonl."
        )

    # -------------------------
    # 2) 출력 경로
    # -------------------------
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

    # -------------------------
    # 3) Parse & collect examples
    # -------------------------
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
