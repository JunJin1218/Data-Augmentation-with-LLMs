"""
Grok "retrieve" helper.

The original OpenAI `gpt/retreive.py` retrieves completed batch jobs by
batch_id and downloads the output file. Grok (xAI) flow in this repo uses a
per-request synchronous generator (`grok_generate.py`) which writes
`batchoutput_*.jsonl` directly into `data/{task}/{model}/`.

This helper fills the role of a "retrieve" step for Grok by doing two things:
 - If any `batchoutput_*.jsonl` exist under Hydra `outputs/...`, copy them into
   the expected `data/{task}/{model}` directory.
 - If no such files are found, it reports the absence so you can run
   `grok_generate.py` to produce them.

Usage:
  # use .env for XAI_API_KEY if needed
  python grok/grok_retrieve.py

Options:
  --hydra-run-dir PATH    path to a hydra outputs run folder (optional)
  --task TASK             override task name (default from gpt/setting.yaml)
  --model MODEL           override model name (default from gpt/setting.yaml)
"""

import os
import shutil
from pathlib import Path
import argparse
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def read_setting_task_model():
    cfg_path = Path("gpt") / "setting.yaml"
    task = None
    model = None
    if cfg_path.exists():
        try:
            import yaml
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                dataset = cfg.get("dataset")
                subset = cfg.get("subset")
                model = cfg.get("model")
                if dataset and subset:
                    task = f"{dataset}-{subset}"
        except Exception:
            pass
    return task, model


def find_hydra_batchoutputs(hydra_root: Path):
    # find any batchoutput_*.jsonl under hydra_root
    if not hydra_root.exists():
        return []
    return list(hydra_root.rglob("batchoutput_*.jsonl"))


def copy_to_data(found_paths, task, model):
    dest_base = Path("data") / task / model
    dest_base.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in sorted(found_paths):
        dest = dest_base / p.name
        shutil.copy2(p, dest)
        copied.append(dest)
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydra-run-dir", type=str, default=None, help="Path to hydra run outputs folder (optional)")
    parser.add_argument("--task", type=str, default=None, help="Task name e.g. super_glue-cb")
    parser.add_argument("--model", type=str, default=None, help="Model name e.g. grok-4-latest")
    args = parser.parse_args()

    task, model = read_setting_task_model()
    if args.task:
        task = args.task
    if args.model:
        model = args.model

    if not task:
        print("[WARN] Could not infer task from gpt/setting.yaml. Use --task to set it.")
        task = "super_glue-cb"
    if not model:
        model = os.environ.get("XAI_MODEL", "grok-4-latest")

    # 1) If hydra-run-dir provided, look there for batchoutput files
    candidates = []
    if args.hydra_run_dir:
        hydra_path = Path(args.hydra_run_dir)
        candidates = find_hydra_batchoutputs(hydra_path)
        if not candidates:
            print(f"[INFO] No batchoutput_*.jsonl files found under {hydra_path}")
    else:
        # try to find any under outputs/*
        outputs_root = Path("outputs")
        if outputs_root.exists():
            # search all run dirs
            for run_dir in outputs_root.iterdir():
                # standard hydra run structure puts files under run_dir/gpt/batches/<task> or directly under run_dir
                candidates.extend(find_hydra_batchoutputs(run_dir))

    if not candidates:
        print(f"[INFO] No Hydra-produced batchoutput files found. Check 'data/{task}/{model}' or run grok/grok_generate.py to produce outputs.")
        # also check data dir for existing outputs
        data_dir = Path("data") / task / model
        if data_dir.exists():
            found = list(data_dir.glob("batchoutput_*.jsonl"))
            if found:
                print(f"[INFO] Existing batchoutput files already in {data_dir}:")
                for f in found:
                    print("  -", f)
                print("[OK] Nothing to copy.")
                return
        return

    # copy candidates into data/{task}/{model}
    copied = copy_to_data(candidates, task, model)
    print(f"[OK] Copied {len(copied)} files into data/{task}/{model}:")
    for c in copied:
        print("  -", c)


if __name__ == '__main__':
    main()
