"""
Simple Grok batch runner.

This script reads batch input files produced by `gpt/prompt.py` (JSONL records),
sends each record as a request to the Grok (xAI) chat endpoint, and writes a
`batchoutput_XXXX.jsonl` file under `data/{task}/{model}/` for each input batch
file. The output format is shaped so `gpt/convert_to_dataset.py` can parse it.

Usage:
    # ensure XAI_API_KEY is set in .env or environment
    python gpt/grok_generate.py

You can override the `output_dir` where batchinput files live (for example if
Hydra wrote them under `outputs/...`) with `output_dir=ABS_PATH`.
"""

import os
import json
import time
from pathlib import Path
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import requests

# Make sure `gpt/prompts` is importable when running this script from the repo root
import sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[1]
_gpt_path = str(_repo_root / "gpt")
if _gpt_path not in sys.path:
    sys.path.insert(0, _gpt_path)

from prompts.utils import get_task_name


def get_sorted_batch_files(batch_input_dir: str):
    p = Path(batch_input_dir)
    if not p.exists():
        return []
    files = [f.name for f in p.iterdir() if f.name.startswith("batchinput_") and f.name.endswith(".jsonl")]
    return sorted(files)


def run_batch_file(batch_input_dir: str, fname: str, out_dir: Path, api_key: str):
    in_path = Path(batch_input_dir) / fname
    out_path = out_dir / f"batchoutput_{int(fname.split('_')[1].split('.')[0]):04d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    with in_path.open("r", encoding="utf-8") as in_f, out_path.open("w", encoding="utf-8") as out_f:
        for line_idx, line in enumerate(in_f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON in {fname} line {line_idx}")
                continue

            body = record.get("body") or record.get("request") or {}

            # Build Grok-compatible payload. If prompt records use `input` with role-structured messages,
            # pass them as `messages`. Otherwise, try to send the body as-is.
            model = body.get("model", os.environ.get("XAI_MODEL", "grok-4-latest"))
            payload = {}
            if "input" in body and isinstance(body["input"], list):
                # map OpenAI-style `input` (list of {'role','content'}) to Grok `messages`
                payload = {
                    "model": model,
                    "messages": body["input"],
                }
            else:
                # fallback — send entire body and include model
                payload = dict(body)
                payload.setdefault("model", model)

            try:
                resp = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Request failed for {fname} line {line_idx}: {e}")
                # write a placeholder error entry so convert step can skip it
                entry = {"response": {"body": {"output": [{"content": [{"type": "output_text", "text": ""}] }] }}, "error": str(e)}
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            if resp.status_code != 200:
                print(f"[WARN] Non-200 from Grok for {fname} line {line_idx}: {resp.status_code} {resp.text}")
                # still write something for downstream parsing
                entry = {"response": {"body": {"output": [{"content": [{"type": "output_text", "text": ""}] }] }}, "status": resp.status_code, "raw": resp.text}
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            try:
                data = resp.json()
            except ValueError:
                print(f"[WARN] Response was not JSON for {fname} line {line_idx}")
                entry = {"response": {"body": {"output": [{"content": [{"type": "output_text", "text": resp.text}] }] }}}
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            # Extract text from common Grok response structures.
            # Grok Chat completions often include `choices[0].message.content` or similar.
            generated_text = ""
            if isinstance(data, dict):
                # check a few common locations
                if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                    first = data["choices"][0]
                    # try `message.content` or `text`
                    if isinstance(first, dict):
                        msg = first.get("message") or first
                        if isinstance(msg, dict):
                            generated_text = msg.get("content") or msg.get("text") or ""
                # fallback: check top-level 'output' or 'response' keys
                if not generated_text:
                    generated_text = data.get("output") or data.get("response") or ""

            # Ensure string
            if isinstance(generated_text, (list, dict)):
                generated_text = json.dumps(generated_text, ensure_ascii=False)
            else:
                generated_text = str(generated_text)

            # Build an entry shaped like OpenAI batch outputs expected by convert_to_dataset
            entry = {
                "response": {
                    "body": {
                        "output": [
                            {
                                "content": [
                                    {"type": "output_text", "text": generated_text}
                                ]
                            }
                        ]
                    }
                }
            }

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Be polite with API rate limits
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None, help="Path to batchinput folder (overrides default gpt/batches/<task>)")
    args = parser.parse_args()

    # load config from gpt/setting.yaml if present to derive task/model
    # but we can simply inspect the batchinput path name for task
    # default: gpt/batches/{task}
    default_task = None
    # Try to read model from env or fallback
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set in environment or .env")

    # If an explicit output_dir specified, use it; else use gpt/batches/{task} where task derived from gpt/setting.yaml
    if args.output_dir:
        batch_input_dir = args.output_dir
        # attempt to infer task name from path
        try:
            default_task = Path(batch_input_dir).name
        except Exception:
            default_task = "unknown"
    else:
        # try to read cfg via prompts.utils get_task_name requires a cfg; but simpler: read gpt/setting.yaml
        cfg_path = Path("gpt") / "setting.yaml"
        task = None
        model = os.environ.get("XAI_MODEL", "grok-4-latest")
        if cfg_path.exists():
            try:
                import yaml
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    dataset = cfg.get("dataset")
                    subset = cfg.get("subset")
                    model = cfg.get("model", model)
                    if dataset and subset:
                        task = f"{dataset}-{subset}"
            except Exception:
                pass

        if not task:
            task = "super_glue-cb"

        batch_input_dir = str(Path("gpt") / "batches" / task)

    # Determine model name for output dir structure
    # try to read model from gpt/setting.yaml if available
    model = os.environ.get("XAI_MODEL", "grok-4-latest")
    cfg_path = Path("gpt") / "setting.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                model = cfg.get("model", model)
        except Exception:
            pass

    batch_files = get_sorted_batch_files(batch_input_dir)
    if not batch_files:
        print(f"[INFO] No batchinput_*.jsonl files found in: {batch_input_dir}")
        return

    out_base = Path("data") / (default_task or Path(batch_input_dir).name) / model
    out_base.mkdir(parents=True, exist_ok=True)

    for fname in batch_files:
        print(f"[INFO] Processing {fname} -> output dir {out_base}")
        run_batch_file(batch_input_dir, fname, out_base, api_key)

    print("[OK] Grok generation finished.")


if __name__ == "__main__":
    main()
