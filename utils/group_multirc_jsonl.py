import json
import sys
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

"""
Group a flattened MultiRC synthetic.jsonl (one line per answer) into one line per (passage, question),
with answers collected into a list of {"text": ..., "label": 0|1}.

Input format (per line):
{"passage": str, "question": str, "answer": str, "label": int}

Output format (per line):
{
  "passage": str,
  "question": str,
  "answers": [ {"text": str, "label": int}, ... ],
  "num_answers": int
}

Usage:
python utils/group_multirc_jsonl.py <input_jsonl> <output_jsonl>
"""

def group_multirc(input_path: str, output_path: str) -> None:
    groups: "OrderedDict[Tuple[str, str], List[Tuple[str, int]]]" = OrderedDict()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # skip bad lines
                continue
            passage = obj.get("passage")
            question = obj.get("question")
            answer = obj.get("answer")
            label = obj.get("label")
            if not isinstance(passage, str) or not isinstance(question, str) or not isinstance(answer, str):
                continue
            try:
                label_int = int(label)
            except Exception:
                label_int = 0
            key = (passage, question)
            if key not in groups:
                groups[key] = []
            groups[key].append((answer, label_int))

    with open(output_path, "w", encoding="utf-8") as out:
        for (passage, question), ans_list in groups.items():
            rec = {
                "passage": passage,
                "question": question,
                "answers": [{"text": a, "label": l} for a, l in ans_list],
                "num_answers": len(ans_list),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python utils/group_multirc_jsonl.py <input_jsonl> <output_jsonl>")
        sys.exit(1)
    group_multirc(sys.argv[1], sys.argv[2])
