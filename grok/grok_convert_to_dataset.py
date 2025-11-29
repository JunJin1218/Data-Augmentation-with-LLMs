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

    # --- helper: try to parse a string as JSON and return a list of canonical pairs ---
    def normalize_pair_obj(o: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(o, dict):
            return None

        # case-insensitive key lookup
        def get_ci(d: Dict[str, Any], key: str):
            for k, v in d.items():
                if k.lower() == key.lower():
                    return v
            return None

        prem = get_ci(o, "premise")
        hyp = get_ci(o, "hypothesis")
        lab = get_ci(o, "label")

        if isinstance(prem, str) and isinstance(hyp, str) and (isinstance(lab, int) or isinstance(lab, float) or (isinstance(lab, str) and lab.isdigit())):
            try:
                lab_int = int(lab)
            except Exception:
                return None
            return {"Premise": prem.strip(), "Hypothesis": hyp.strip(), "Label": int(lab_int)}
        return None

    def parse_candidate(s: str) -> Optional[List[Dict[str, Any]]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None

        # Case 1: dict with a 'pairs' key (case-insensitive)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == "pairs" and isinstance(v, list):
                    out: List[Dict[str, Any]] = []
                    for item in v:
                        np = normalize_pair_obj(item)
                        if np is not None:
                            out.append(np)
                    return out if out else None

            # Case: single object that itself contains premise/hypothesis/label
            single = normalize_pair_obj(obj)
            if single is not None:
                return [single]

        # Case 2: top-level list [ {...}, {...}, ... ]
        if isinstance(obj, list):
            out: List[Dict[str, Any]] = []
            for item in obj:
                np = normalize_pair_obj(item)
                if np is not None:
                    out.append(np)
            return out if out else None

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

    # --- 3) Embedded JSON objects with premise/hypothesis/label anywhere ---
    # Find each { ... } that mentions those keys in some order (case-insensitive)
    snippets = re.findall(
        r'\{[^{}]*"premise"[^{}]*"hypothesis"[^{}]*"label"[^{}]*\}',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    parsed_pairs: List[Dict[str, Any]] = []
    for snip in snippets:
        # Remove '//' comments inside the object (Grok sometimes adds explanations)
        snippet_no_comments = re.sub(r'//.*', '', snip)
        try:
            obj = json.loads(snippet_no_comments)
        except json.JSONDecodeError:
            continue
        np = normalize_pair_obj(obj)
        if np is not None:
            parsed_pairs.append(np)

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


def extract_copa_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse COPA-style outputs into a list of dicts with keys:
    - "premise", "choice1", "choice2", "question", "label" (0/1)

    Accepts: whole-text JSON, fenced JSON, embedded objects, or simple labeled lines.
    """
    text = text.strip()
    if not text:
        return None

    def get_ci(d: Dict[str, Any], key: str):
        for k, v in d.items():
            if k.lower() == key.lower():
                return v
        return None

    def norm_item(o: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(o, dict):
            return None
        premise = get_ci(o, "premise")
        c1 = get_ci(o, "choice1")
        c2 = get_ci(o, "choice2")
        question = get_ci(o, "question")
        label = get_ci(o, "label")
        if not (isinstance(premise, str) and isinstance(c1, str) and isinstance(c2, str) and isinstance(question, str)):
            return None
        if isinstance(label, (int, float)):
            lab = int(label)
        elif isinstance(label, str) and label.strip().isdigit():
            lab = int(label.strip())
        else:
            return None
        question = question.strip().lower()
        if question not in ("cause", "effect"):
            return None
        return {
            "premise": premise.strip(),
            "choice1": c1.strip(),
            "choice2": c2.strip(),
            "question": question,
            "label": lab,
        }

    def try_json(s: str) -> Optional[List[Dict[str, Any]]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            # common: {"items": [...]} or top-level single item
            for k, v in obj.items():
                if k.lower() in ("items", "pairs", "examples") and isinstance(v, list):
                    out = []
                    for it in v:
                        ni = norm_item(it)
                        if ni:
                            out.append(ni)
                    return out or None
            single = norm_item(obj)
            return [single] if single else None
        if isinstance(obj, list):
            out = []
            for it in obj:
                ni = norm_item(it)
                if ni:
                    out.append(ni)
            return out or None
        return None

    # 1) direct JSON
    items = try_json(text)
    if items:
        return items

    # 2) fenced blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json"):
                part = part[4:].strip()
            items = try_json(part)
            if items:
                return items

    # 3) embedded objects with required keys
    snippets = re.findall(
        r'\{[^{}]*"premise"[^{}]*"choice1"[^{}]*"choice2"[^{}]*"question"[^{}]*"label"[^{}]*\}',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out: List[Dict[str, Any]] = []
    for snip in snippets:
        snip_nc = re.sub(r'//.*', '', snip)
        try:
            obj = json.loads(snip_nc)
        except json.JSONDecodeError:
            continue
        ni = norm_item(obj)
        if ni:
            out.append(ni)
    if out:
        return out

    # 4) labeled lines fallback
    pat = re.compile(
        r'Premise:\s*(.+?)\s*Choice1:\s*(.+?)\s*Choice2:\s*(.+?)\s*Question:\s*(cause|effect)\s*Label:\s*([01])',
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = []
    for prem, c1, c2, q, lab in pat.findall(text):
        out.append({
            "premise": prem.strip(),
            "choice1": c1.strip(),
            "choice2": c2.strip(),
            "question": q.strip().lower(),
            "label": int(lab),
        })
    return out or None


def extract_wsc_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse WSC-style outputs into a list of dicts with keys:
    - "text", "span1_text" (pronoun), "span2_text" (candidate), optional indices, and "label" (0/1)
    """
    text = text.strip()
    if not text:
        return None

    def get_ci(d: Dict[str, Any], key: str):
        for k, v in d.items():
            if k.lower() == key.lower():
                return v
        return None

    def norm_item(o: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(o, dict):
            return None
        t = get_ci(o, "text")
        s1 = get_ci(o, "span1_text") or get_ci(o, "span1text")
        s2 = get_ci(o, "span2_text") or get_ci(o, "span2text")
        lab = get_ci(o, "label")
        s1i = get_ci(o, "span1_index")
        s2i = get_ci(o, "span2_index")
        if not (isinstance(t, str) and isinstance(s1, str) and isinstance(s2, str)):
            return None
        if isinstance(lab, (int, float)):
            lab = int(lab)
        elif isinstance(lab, str) and lab.strip().isdigit():
            lab = int(lab.strip())
        else:
            return None
        out = {
            "text": t.strip(),
            "span1_text": s1.strip(),
            "span2_text": s2.strip(),
            "label": lab,
        }
        if isinstance(s1i, (int, float)):
            out["span1_index"] = int(s1i)
        if isinstance(s2i, (int, float)):
            out["span2_index"] = int(s2i)
        return out

    def try_json(s: str) -> Optional[List[Dict[str, Any]]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in ("items", "pairs", "examples") and isinstance(v, list):
                    out = [ni for it in v if (ni := norm_item(it))]
                    return out or None
            single = norm_item(obj)
            return [single] if single else None
        if isinstance(obj, list):
            out = [ni for it in obj if (ni := norm_item(it))]
            return out or None
        return None

    items = try_json(text)
    if items:
        return items
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json"):
                part = part[4:].strip()
            items = try_json(part)
            if items:
                return items

    snippets = re.findall(
        r'\{[^{}]*"text"[^{}]*"span1(?:_?text)"[^{}]*"span2(?:_?text)"[^{}]*"label"[^{}]*\}',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out: List[Dict[str, Any]] = []
    for snip in snippets:
        snip_nc = re.sub(r'//.*', '', snip)
        try:
            obj = json.loads(snip_nc)
        except json.JSONDecodeError:
            continue
        ni = norm_item(obj)
        if ni:
            out.append(ni)
    if out:
        return out

    pat = re.compile(
        r'Text:\s*(.+?)\s*Span1Text:\s*(.+?)\s*Span2Text:\s*(.+?)\s*Label:\s*([01])',
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = []
    for t, s1, s2, lab in pat.findall(text):
        out.append({
            "text": t.strip(),
            "span1_text": s1.strip(),
            "span2_text": s2.strip(),
            "label": int(lab),
        })
    return out or None


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


def convert_super_glue_copa(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for it in items:
        prem = it.get("premise")
        c1 = it.get("choice1")
        c2 = it.get("choice2")
        q = (it.get("question") or "").lower()
        lab = it.get("label")
        if not (prem and c1 and c2 and q in ("cause", "effect") and lab is not None):
            continue
        examples.append({
            "premise": prem,
            "choice1": c1,
            "choice2": c2,
            "question": q,
            "label": int(lab),
        })
    return examples


def convert_super_glue_wsc(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for it in items:
        text = it.get("text")
        s1 = it.get("span1_text")
        s2 = it.get("span2_text")
        lab = it.get("label")
        if text is None or s1 is None or s2 is None or lab is None:
            continue
        ex: Dict[str, Any] = {
            "text": text,
            "span1_text": s1,
            "span2_text": s2,
            "label": int(lab),
        }
        # Try to infer indices if missing
        if "span1_index" in it and isinstance(it["span1_index"], int):
            ex["span1_index"] = it["span1_index"]
        else:
            idx = text.lower().find(str(s1).lower())
            ex["span1_index"] = idx if idx >= 0 else -1
        if "span2_index" in it and isinstance(it["span2_index"], int):
            ex["span2_index"] = it["span2_index"]
        else:
            idx = text.lower().find(str(s2).lower())
            ex["span2_index"] = idx if idx >= 0 else -1
        examples.append(ex)
    return examples


def extract_wic_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse WiC-style outputs into a list with keys:
    - "word", "sentence1", "sentence2", "label" (0/1)
    """
    text = text.strip()
    if not text:
        return None

    def get_ci(d: Dict[str, Any], key: str):
        for k, v in d.items():
            if k.lower() == key.lower():
                return v
        return None

    def to_int_label(v: Any) -> Optional[int]:
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        if isinstance(v, bool):
            return 1 if v else 0
        return None

    def norm_item(o: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(o, dict):
            return None
        word = get_ci(o, "word")
        s1 = get_ci(o, "sentence1")
        s2 = get_ci(o, "sentence2")
        lab = to_int_label(get_ci(o, "label"))
        if isinstance(word, str) and isinstance(s1, str) and isinstance(s2, str) and lab is not None:
            return {"word": word.strip(), "sentence1": s1.strip(), "sentence2": s2.strip(), "label": lab}
        return None

    def try_json(s: str) -> Optional[List[Dict[str, Any]]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in ("items", "pairs", "examples") and isinstance(v, list):
                    out = [ni for it in v if (ni := norm_item(it))]
                    return out or None
            one = norm_item(obj)
            return [one] if one else None
        if isinstance(obj, list):
            out = [ni for it in obj if (ni := norm_item(it))]
            return out or None
        return None

    items = try_json(text)
    if items:
        return items
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json"):
                part = part[4:].strip()
            items = try_json(part)
            if items:
                return items

    # Embedded objects
    snippets = re.findall(
        r'\{[^{}]*"word"[^{}]*"sentence1"[^{}]*"sentence2"[^{}]*"label"[^{}]*\}',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out: List[Dict[str, Any]] = []
    for snip in snippets:
        snip_nc = re.sub(r'//.*', '', snip)
        try:
            obj = json.loads(snip_nc)
        except json.JSONDecodeError:
            continue
        ni = norm_item(obj)
        if ni:
            out.append(ni)
    if out:
        return out

    # labeled lines
    pat = re.compile(
        r'Word:\s*(.+?)\s*Sentence1:\s*(.+?)\s*Sentence2:\s*(.+?)\s*Label:\s*([01])',
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = []
    for w, s1, s2, lab in pat.findall(text):
        out.append({"word": w.strip(), "sentence1": s1.strip(), "sentence2": s2.strip(), "label": int(lab)})
    return out or None


def extract_boolq_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse BoolQ-style outputs into a list with keys:
    - "passage", "question", "label" (0/1)
    """
    text = text.strip()
    if not text:
        return None

    def get_ci(d: Dict[str, Any], key: str):
        for k, v in d.items():
            if k.lower() == key.lower():
                return v
        return None

    def to_int_label(v: Any) -> Optional[int]:
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        if isinstance(v, bool):
            return 1 if v else 0
        return None

    def norm_item(o: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(o, dict):
            return None
        p = get_ci(o, "passage")
        q = get_ci(o, "question")
        lab = to_int_label(get_ci(o, "label"))
        if isinstance(p, str) and isinstance(q, str) and lab is not None:
            return {"passage": p.strip(), "question": q.strip(), "label": lab}
        return None

    def try_json(s: str) -> Optional[List[Dict[str, Any]]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in ("items", "pairs", "examples") and isinstance(v, list):
                    out = [ni for it in v if (ni := norm_item(it))]
                    return out or None
            one = norm_item(obj)
            return [one] if one else None
        if isinstance(obj, list):
            out = [ni for it in obj if (ni := norm_item(it))]
            return out or None
        return None

    items = try_json(text)
    if items:
        return items
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json"):
                part = part[4:].strip()
            items = try_json(part)
            if items:
                return items

    snippets = re.findall(
        r'\{[^{}]*"passage"[^{}]*"question"[^{}]*"label"[^{}]*\}',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out: List[Dict[str, Any]] = []
    for snip in snippets:
        snip_nc = re.sub(r'//.*', '', snip)
        try:
            obj = json.loads(snip_nc)
        except json.JSONDecodeError:
            continue
        ni = norm_item(obj)
        if ni:
            out.append(ni)
    if out:
        return out

    pat = re.compile(
        r'Passage:\s*(.+?)\s*Question:\s*(.+?)\s*Label:\s*([01])',
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = []
    for p, q, lab in pat.findall(text):
        out.append({"passage": p.strip(), "question": q.strip(), "label": int(lab)})
    return out or None


def extract_multirc_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse MultiRC-style outputs.

    Supports TWO output styles and normalizes to a flat list of items:
    1) Aggregated answers per question:
        {"passage": ..., "question": ..., "answers": [{"text": ..., "label": 0/1}, ...]}
        → expands to multiple items with keys: "passage", "question", "answer", "label".

    2) One-answer-per-line JSONL (preferred):
        {"passage": ..., "question": ..., "answer": ..., "label": 0/1}
        → passes through as a single normalized item.
     """
    text = text.strip()
    if not text:
        return None

    def get_ci(d: Dict[str, Any], key: str):
        for k, v in d.items():
            if k.lower() == key.lower():
                return v
        return None

    def to_int_label(v: Any) -> Optional[int]:
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        if isinstance(v, bool):
            return 1 if v else 0
        return None

    def norm_answer(a: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(a, dict):
            return None
        t = get_ci(a, "text")
        lab = to_int_label(get_ci(a, "label"))
        if isinstance(t, str) and lab is not None:
            return {"text": t.strip(), "label": lab}
        return None

    def norm_item(o: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(o, dict):
            return None
        p = get_ci(o, "passage") or get_ci(o, "paragraph")
        q = get_ci(o, "question")

        # Style (2): one-answer-per-line
        a_single = get_ci(o, "answer") or get_ci(o, "ans")
        lab_single = to_int_label(get_ci(o, "label"))
        if isinstance(p, str) and isinstance(q, str) and isinstance(a_single, str) and lab_single is not None:
            return [{"passage": p.strip(), "question": q.strip(), "answer": a_single.strip(), "label": lab_single}]

        # Style (1): aggregated answers
        ans = get_ci(o, "answers")
        if isinstance(p, str) and isinstance(q, str) and isinstance(ans, list):
            normed = [na for it in ans if (na := norm_answer(it))]
            if normed:
                return [
                    {"passage": p.strip(), "question": q.strip(), "answer": a["text"], "label": a["label"]}
                    for a in normed
                ]
        return None

    def try_json(s: str) -> Optional[List[Dict[str, Any]]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in ("items", "pairs", "examples") and isinstance(v, list):
                    out: List[Dict[str, Any]] = []
                    for it in v:
                        ni = norm_item(it)
                        if ni:
                            out.extend(ni)
                    return out or None
            one = norm_item(obj)
            if one:
                return one
        if isinstance(obj, list):
            out: List[Dict[str, Any]] = []
            for it in obj:
                ni = norm_item(it)
                if ni:
                    out.extend(ni)
            return out or None
        return None

    items = try_json(text)
    if items:
        return items

    # Try JSONL (line-delimited JSON)
    def try_jsonl(s: str) -> Optional[List[Dict[str, Any]]]:
        lines = s.strip().split('\n')
        out: List[Dict[str, Any]] = []
        valid_lines = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ni = norm_item(obj)
                if ni:
                    out.extend(ni)
                    valid_lines += 1
            except json.JSONDecodeError:
                pass
        if valid_lines > 0:
            return out
        return None

    items = try_jsonl(text)
    if items:
        return items

    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json"):
                part = part[4:].strip()
            items = try_json(part)
            if items:
                return items
            items = try_jsonl(part)
            if items:
                return items

    snippets = re.findall(
        r'\{[^{}]*"(passage|paragraph)"[^{}]*"question"[^{}]*(?:"answers"|"answer")[^{}]*\}',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out: List[Dict[str, Any]] = []
    for snip in snippets:
        snip_nc = re.sub(r'//.*', '', snip)
        try:
            obj = json.loads(snip_nc)
        except json.JSONDecodeError:
            continue
        ni = norm_item(obj)
        if ni:
            out.extend(ni)
    if out:
        return out

    # Fallback: capitalized single-answer objects or arrays under 'pairs'
    def norm_cap_item(o: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(o, dict):
            return None
        def get_ci_local(d, key):
            for k, v in d.items():
                if k.lower() == key.lower():
                    return v
            return None
        p = get_ci_local(o, "passage") or get_ci_local(o, "paragraph")
        q = get_ci_local(o, "question")
        a = get_ci_local(o, "answer")
        lab = to_int_label(get_ci_local(o, "label"))
        if isinstance(p, str) and isinstance(q, str) and isinstance(a, str) and lab is not None:
            return {"passage": p.strip(), "question": q.strip(), "answer": a.strip(), "label": lab}
        # Also accept capitalized keys: Passage, Question, Answer, Label
        p = o.get("Passage") or o.get("Paragraph") or p
        q = o.get("Question") or q
        a = o.get("Answer") or a
        lab_raw = o.get("Label")
        lab2 = to_int_label(lab_raw)
        if isinstance(p, str) and isinstance(q, str) and isinstance(a, str) and lab2 is not None:
            return {"passage": p.strip(), "question": q.strip(), "answer": a.strip(), "label": lab2}
        return None

    # Try to parse any top-level JSON with 'pairs' containing capitalized items
    cap_pairs = None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == "pairs" and isinstance(v, list):
                    tmp: List[Dict[str, Any]] = []
                    for it in v:
                        ni = norm_cap_item(it)
                        if ni:
                            tmp.append(ni)
                    cap_pairs = tmp or None
    except Exception:
        cap_pairs = None
    if cap_pairs:
        return cap_pairs

    # Labeled lines fallback (coarse)
    # Passage: ... Question: ... Answers:
    # - <text> | Label: 0/1
    # - <text> | Label: 0/1
    blk_pat = re.compile(
        r'Passage:\s*(.+?)\s*Question:\s*(.+?)\s*Answers:\s*((?:.-.*?\n?)+)',
        flags=re.DOTALL | re.IGNORECASE,
    )
    ans_pat = re.compile(r'-\s*(.+?)\s*\|\s*Label:\s*([01])', flags=re.IGNORECASE)
    out = []
    for psg, qst, ans_block in blk_pat.findall(text):
        answers = []
        for t, lab in ans_pat.findall(ans_block):
            answers.append({"text": t.strip(), "label": int(lab)})
        if answers:
            # expand to flat items
            out.extend([
                {"passage": psg.strip(), "question": qst.strip(), "answer": a["text"], "label": a["label"]}
                for a in answers
            ])
    return out or None


def extract_record_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse ReCoRD-style outputs into a list with keys:
    - "passage", "query", "entities", "answers"
    """
    text = text.strip()
    if not text:
        return None

    def get_ci(d: Dict[str, Any], key: str):
        for k, v in d.items():
            if k.lower() == key.lower():
                return v
        return None

    def norm_item(o: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(o, dict):
            return None
        p = get_ci(o, "passage")
        q = get_ci(o, "query")
        ents = get_ci(o, "entities")
        ans = get_ci(o, "answers")
        
        if isinstance(p, str) and isinstance(q, str) and isinstance(ents, list) and isinstance(ans, list):
            return {"passage": p.strip(), "query": q.strip(), "entities": ents, "answers": ans}
        return None

    def try_json(s: str) -> Optional[List[Dict[str, Any]]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in ("items", "pairs", "examples") and isinstance(v, list):
                    out = [ni for it in v if (ni := norm_item(it))]
                    return out or None
            one = norm_item(obj)
            return [one] if one else None
        if isinstance(obj, list):
            out = [ni for it in obj if (ni := norm_item(it))]
            return out or None
        return None

    items = try_json(text)
    if items:
        return items
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json"):
                part = part[4:].strip()
            items = try_json(part)
            if items:
                return items

    snippets = re.findall(
        r'\{[^{}]*"passage"[^{}]*"query"[^{}]*"entities"[^{}]*"answers"[^{}]*\}',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out: List[Dict[str, Any]] = []
    for snip in snippets:
        snip_nc = re.sub(r'//.*', '', snip)
        try:
            obj = json.loads(snip_nc)
        except json.JSONDecodeError:
            continue
        ni = norm_item(obj)
        if ni:
            out.append(ni)
    if out:
        return out

    return None


TASK_CONVERTERS: Dict[str, Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = {
    "glue-mrpc": convert_glue_mrpc,
    "super_glue-cb": convert_super_glue_cb,
    # Additions for other SuperGLUE tasks
    # "super_glue-rte": convert_super_glue_rte,
    "super_glue-rte": convert_super_glue_cb,
    "super_glue-copa": convert_super_glue_copa,
    "super_glue-wsc": convert_super_glue_wsc,
    "super_glue-wic": lambda items: items,  # items already in final shape
    "super_glue-boolq": lambda items: items,  # items already in final shape
    "super_glue-multirc": lambda items: items,  # items already normalized to final shape
    "super_glue-record": lambda items: items,   # items already normalized to final shape
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

            # Select parser by task
            if "copa" in task:
                pairs = extract_copa_from_text(text)
            elif "wsc" in task:
                pairs = extract_wsc_from_text(text)
            elif "wic" in task:
                pairs = extract_wic_from_text(text)
            elif "boolq" in task:
                pairs = extract_boolq_from_text(text)
            elif "multirc" in task:
                pairs = extract_multirc_from_text(text)
            elif "record" in task:
                pairs = extract_record_from_text(text)
            else:
                pairs = extract_pairs_from_text(text)
            # Common check for all tasks
            if not pairs:
                snippet = text[:200].replace("\n", " ")
                print(
                    f"[WARN] {path.name} Line {line_idx}: parsing failed for task '{task}'. Snippet: {snippet}"
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
            - input:           data/{task_name}/{model_name}/batchoutput_*.jsonl
            - output (legacy): data/{task_name}/{model_name}/synthetic.jsonl
            - output (new):    data/{task_name}/{model_name}/generated/synthetic.jsonl
    """
    task = get_task_name(cfg)       # e.g. "super_glue-cb"
    model_name = cfg.model          # e.g. "grok-3-mini"

    # Support both layouts; prefer new model-first if present
    candidate_new = Path(to_absolute_path(f"data/{model_name}/{task}"))
    candidate_old = Path(to_absolute_path(f"data/{task}/{model_name}"))
    if candidate_new.exists():
        base_dir = candidate_new
        layout = "model-first"
    elif candidate_old.exists():
        base_dir = candidate_old
        layout = "task-first (legacy)"
    else:
        raise FileNotFoundError(
            "Input dir not found in either layout:\n"
            f" - {candidate_new}\n"
            f" - {candidate_old}\n"
            "Expected batch outputs at .../batchoutput_*.jsonl"
        )

    response_paths = sorted(base_dir.glob("batchoutput_*.jsonl"))

    if not response_paths:
        raise FileNotFoundError(
            f"No batchoutput_*.jsonl files found in {base_dir}.\n"
            f"Check that retrieve script saved files as batchoutput_XXXX.jsonl."
        )

    # Legacy output location
    output_path = base_dir / "synthetic.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # New generated/ subfolder to group final datasets
    generated_dir = base_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_path_generated = generated_dir / "synthetic.jsonl"

    print(f"[INFO] Task        : {task}")
    print(f"[INFO] Model       : {model_name}")
    print(f"[INFO] Input dir   : {base_dir} ({layout})")
    print(f"[INFO] Num inputs  : {len(response_paths)}")
    print(f"[INFO] Input files :")
    for p in response_paths:
        print(f"  - {p.name}")
    print(f"[INFO] Output file (legacy) : {output_path}")
    print(f"[INFO] Output file (new)    : {output_path_generated}")

    all_examples: List[Dict[str, Any]] = []
    for path in response_paths:
        examples = load_pairs_from_batch_output(path, task)
        print(f"[INFO] {path.name}: collected {len(examples)} examples.")
        all_examples.extend(examples)

    print(f"[INFO] Total collected {len(all_examples)} examples. Writing JSONL...")

    # Write to both locations for backward compatibility
    for target in (output_path, output_path_generated):
        with target.open("w", encoding="utf-8") as out_f:
            for ex in all_examples:
                out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
