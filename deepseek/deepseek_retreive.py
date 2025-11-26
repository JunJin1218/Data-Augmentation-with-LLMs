#!/usr/bin/env python
from pathlib import Path
import json

import hydra
from omegaconf import DictConfig

from prompts.utils import get_task_name


def convert_deepseek_outputs_to_batch_outputs(cfg: DictConfig) -> None:
    """
    Read DeepSeek generation outputs from:

        deepseek/outputs/{task}/deepseek_output_*.jsonl

    and convert them into OpenAI-batch-like output files:

        data/{task}/{cfg.model}/batchoutput_*.jsonl

    so that deepseek_convert_to_dataset.py (which expects OpenAI batch output shape)
    can run unchanged.
    """
    task = get_task_name(cfg)
    model_name = cfg.model

    # Where deepseek_generate.py wrote its outputs
    default_deepseek_out_dir = Path(f"deepseek/outputs/{task}")
    deepseek_out_dir = Path(cfg.get("deepseek_output_dir", default_deepseek_out_dir))

    if not deepseek_out_dir.is_dir():
        raise FileNotFoundError(
            f"DeepSeek output dir not found: {deepseek_out_dir}\n"
            f"Expected files like deepseek_output_batchinput_0001.jsonl there."
        )

    # Where deepseek_convert_to_dataset.py expects batch outputs
    output_dir = Path(f"data/{task}/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Task          : {task}")
    print(f"[INFO] Model         : {model_name}")
    print(f"[INFO] DeepSeek out  : {deepseek_out_dir}")
    print(f"[INFO] Target out dir: {output_dir}")

    deepseek_files = sorted(deepseek_out_dir.glob("deepseek_output_*.jsonl"))
    if not deepseek_files:
        raise FileNotFoundError(
            f"No deepseek_output_*.jsonl files found in {deepseek_out_dir}.\n"
            f"Run deepseek_generate.py first."
        )

    print("[INFO] Input files:")
    for p in deepseek_files:
        print(f"  - {p.name}")

    for idx, in_path in enumerate(deepseek_files, start=1):
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

                deepseek_output = entry.get("deepseek_output")
                if not isinstance(deepseek_output, str) or not deepseek_output.strip():
                    print(
                        f"[WARN] {in_path.name}: no 'deepseek_output' text found, skipping line."
                    )
                    continue

                # Wrap DeepSeek output into an OpenAI-batch-like "response.body.output"
                # shape so that extract_output_text() in deepseek_convert_to_dataset.py works.
                batch_style_entry = {
                    "response": {
                        "body": {
                            "output": [
                                {
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": deepseek_output,
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

    print("\n[DONE] All DeepSeek outputs converted to batchoutput_*.jsonl.")


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig) -> None:
    convert_deepseek_outputs_to_batch_outputs(cfg)


if __name__ == "__main__":
    main()
