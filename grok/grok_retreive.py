#!/usr/bin/env python
from pathlib import Path
import json

import hydra
from omegaconf import DictConfig

from prompts.utils import get_task_name


def convert_grok_outputs_to_batch_outputs(cfg: DictConfig) -> None:
    """
    Read grok_generate outputs from:

        grok/outputs/{task}/grok_output_*.jsonl

    and convert them into OpenAI-batch-like output files:

        data/{task}/{cfg.model}/batchoutput_*.jsonl

    so that convert_to_dataset.py (which expects OpenAI batch output shape)
    can run unchanged.
    """
    task = get_task_name(cfg)
    model_name = cfg.model

    # Where grok_generate.py wrote its outputs
    grok_output_dir = Path(f"grok/outputs/{task}")
    if not grok_output_dir.is_dir():
        raise FileNotFoundError(
            f"Grok output dir not found: {grok_output_dir}\n"
            f"Expected files like grok_output_batchinput_0001.jsonl there."
        )

    # Where convert_to_dataset.py expects batchoutput_*.jsonl
    output_dir = Path(f"data/{task}/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Task          : {task}")
    print(f"[INFO] Model         : {model_name}")
    print(f"[INFO] Grok out dir  : {grok_output_dir}")
    print(f"[INFO] Target out dir: {output_dir}")

    # Find all grok_output_*.jsonl files
    grok_files = sorted(grok_output_dir.glob("grok_output_*.jsonl"))
    if not grok_files:
        raise FileNotFoundError(
            f"No grok_output_*.jsonl files found in {grok_output_dir}.\n"
            f"Run grok_generate.py first."
        )

    print("[INFO] Input files:")
    for p in grok_files:
        print(f"  - {p.name}")

    # For each grok_output file, create a corresponding batchoutput_XXXX.jsonl
    for idx, in_path in enumerate(grok_files, start=1):
        out_name = f"batchoutput_{idx:04d}.jsonl"
        out_path = output_dir / out_name

        num_lines_in = 0
        num_lines_out = 0

        with in_path.open("r", encoding="utf-8") as fin, out_path.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                num_lines_in += 1

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] {in_path.name}: invalid JSON ({e}), skipping line.")
                    continue

                grok_output = entry.get("grok_output")
                if not isinstance(grok_output, str) or not grok_output.strip():
                    print(
                        f"[WARN] {in_path.name}: no 'grok_output' text found, skipping line."
                    )
                    continue

                # Wrap Grok output into an OpenAI-batch-like "response.body.output"
                # shape so that extract_output_text() in convert_to_dataset.py works.
                batch_style_entry = {
                    "response": {
                        "body": {
                            "output": [
                                {
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": grok_output,
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }

                fout.write(json.dumps(batch_style_entry, ensure_ascii=False) + "\n")
                num_lines_out += 1

        print(
            f"[OK] {in_path.name} -> {out_name} "
            f"(in: {num_lines_in} lines, out: {num_lines_out} lines)"
        )

    print("\n[DONE] All Grok outputs converted to batchoutput_*.jsonl.")


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig) -> None:
    convert_grok_outputs_to_batch_outputs(cfg)


if __name__ == "__main__":
    main()
