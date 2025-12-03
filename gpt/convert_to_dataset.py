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

# ---- CB ----
def convert_super_glue_cb(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert schema.json 'pairs' to SuperGLUE CB-style examples.
    """
    examples: List[Dict[str, Any]] = []
    for pair in pairs:
        premise = pair.get("premise")
        hypothesis = pair.get("hypothesis")
        label = pair.get("label")
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

# ---- BoolQ ----
def convert_super_glue_boolq(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to BoolQ-style.
    Expected keys per item: passage (str), question (str), label (bool/int)
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        passage = it.get("passage")
        question = it.get("question")
        label = it.get("label")
        if passage is None or question is None or label is None:
            continue
        out.append(
            {
                "passage": passage,
                "question": question,
                "label": int(label),  # ensure 0/1
            }
        )
    return out


# ---- COPA ----
def convert_super_glue_copa(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to COPA-style.
    Expected keys per item: premise, choice1, choice2, question ("cause"|"effect"), label (0|1)
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        premise = it.get("premise")
        c1 = it.get("choice1")
        c2 = it.get("choice2")
        q = it.get("question")
        label = it.get("label")
        if None in (premise, c1, c2, q, label):
            continue
        out.append(
            {
                "premise": premise,
                "choice1": c1,
                "choice2": c2,
                "question": q,
                "label": int(label),
            }
        )
    return out


# ---- MultiRC ----
def convert_super_glue_multirc(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to MultiRC-style.
    Expected keys per item:
      - passage (str)
      - question (str)
      - answer: List[{"text": str, "label": 0|1 or bool}]
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        para = it.get("paragraph")
        q    = it.get("question")
        ans  = it.get("answer")
        lb   = it.get("label")
        if None in (para, q, ans, lb):
            continue
        out.append({
            "paragraph": str(para),
            "question":  str(q),
            "answer":    str(ans),
            "label":     int(lb),
        })
    return out


# ---- RTE ----
def convert_super_glue_rte(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to RTE-style.
    Expected keys per item: premise, hypothesis, label (int; HF RTE는 0/1)
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        premise = it.get("premise")
        hypothesis = it.get("hypothesis")
        label = it.get("label")
        if premise is None or hypothesis is None or label is None:
            continue
        out.append({"premise": premise, "hypothesis": hypothesis, "label": int(label)})
    return out


# ---- WiC ----
def convert_super_glue_wic(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to WiC-style.
    Expected keys per item:
      - sentence1, sentence2, word
      - (optional) start1, end1, start2, end2
      - label (bool/int)
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        s1 = it.get("sentence1")
        s2 = it.get("sentence2")
        word = it.get("word")
        label = it.get("label")
        if s1 is None or s2 is None or word is None or label is None:
            continue
        ex = {
            "sentence1": s1,
            "sentence2": s2,
            "word": word,
            "label": int(label),
        }
        # 위치 정보가 있으면 유지
        for k in ("start1", "end1", "start2", "end2"):
            if k in it:
                ex[k] = it[k]
        out.append(ex)
    return out


# ---- WSC ----
def convert_super_glue_wsc(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to WSC-style (SuperGLUE WSC).
    Expected keys per item:
      - text
      - span1_text, span1_index
      - span2_text, span2_index
      - label (bool/int)
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        text = it.get("text")
        s1t = it.get("span1_text")
        s2t = it.get("span2_text")
        label = it.get("label")
        if None in (text, s1t, s2t, label):
            continue
        out.append(
            {
                "text": text,
                "span1_text": s1t,
                "span2_text": s2t,
                "label": int(label),
            }
        )
    return out


# ---- ReCoRD ----
def convert_super_glue_record(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to ReCoRD-style.
    Expected keys per item:
      - passage (str), query (str with @placeholder), entities (List[str]), answers (List[str])
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        passage = it.get("passage")
        query = it.get("query")
        entities = it.get("entities")
        answers = it.get("answers")
        if passage is None or query is None or not isinstance(entities, list) or not isinstance(answers, list):
            continue
        out.append(
            {
                "passage": passage,
                "query": query,
                "entities": list(entities),
                "answers": list(answers),
            }
        )
    return out

def convert_biosses(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to ReCoRD-style.
    Expected keys per item:
      - passage (str), query (str with @placeholder), entities (List[str]), answers (List[str])
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        s1 = it.get("sentence1")
        s2 = it.get("sentence2")
        score = it.get("score")
        if s1 is None or s2 is None or score is None:
            continue
        out.append(
            {
                "sentence1": s1,
                "sentence2": s2,
                "score": score
            }
        )
    return out

# 여기에 태스크별 컨버터를 계속 추가하면 됨
# def convert_super_glue_rte(...): ...
# def convert_sst2(...): ...


TASK_CONVERTERS: Dict[str, Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = {
    "glue-mrpc": convert_glue_mrpc,
    "super_glue-cb": convert_super_glue_cb,
    "super_glue-boolq": convert_super_glue_boolq,
    "super_glue-copa": convert_super_glue_copa,
    "super_glue-multirc": convert_super_glue_multirc,
    "super_glue-rte": convert_super_glue_rte,
    "super_glue-wic": convert_super_glue_wic,
    "super_glue-wsc": convert_super_glue_wsc,
    "super_glue-record": convert_super_glue_record,
    "biosses": convert_biosses,
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
    if (cfg.dataset not in ["biosses"]):
        task = get_task_name(cfg)
    else:
        task = cfg.dataset
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
