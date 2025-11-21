#!/usr/bin/env python
from dataclasses import dataclass
from pathlib import Path
import json

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from openai import OpenAI

from prompts.utils import get_task_name


def retrieve_batches(cfg) -> None:
    batch_id_list_path = Path(f"gpt/batches/{get_task_name(cfg)}/batch_id_list.jsonl")
    output_dir = Path(f"data/{get_task_name(cfg)}/{cfg.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading batch ids from: {batch_id_list_path}")
    print(f"[INFO] Saving outputs under:   {output_dir}")

    if not batch_id_list_path.is_file():
        raise FileNotFoundError(f"batch_id_list.jsonl not found: {batch_id_list_path}")

    client = OpenAI()  # OPENAI_API_KEY는 env에서 읽힌다고 가정

    with batch_id_list_path.open("r", encoding="utf-8") as f:
        idx = 1
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                mapping = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error on line: {line} ({e})")
                continue

            if not isinstance(mapping, dict) or len(mapping) != 1:
                print(f"[WARN] Unexpected line format (expect 1 key-value): {line}")
                continue

            input_filename, batch_id = next(iter(mapping.items()))
            print(f"\n[INFO] Processing {input_filename} -> {batch_id}")

            # 1) batch retrieve
            try:
                batch = client.batches.retrieve(batch_id)
            except Exception as e:
                print(f"[ERROR] Failed to retrieve batch {batch_id}: {e}")
                continue

            status = getattr(batch, "status", None)
            print(f"[INFO] Batch status: {status}")

            if status != "completed":
                print(f"[WARN] Batch {batch_id} not completed (status={status}), skip.")
                continue

            output_file_id = getattr(batch, "output_file_id", None)
            if not output_file_id:
                print(f"[WARN] Batch {batch_id} has no output_file_id, skip.")
                continue

            # 2) files API로 content 가져오기
            try:
                file_response = client.files.content(output_file_id)
                content = file_response.text  # openai-python 1.x 기준
            except Exception as e:
                print(
                    f"[ERROR] Failed to fetch file content for {output_file_id}: {e}"
                )
                continue

            # 3) batchoutput_XXXX.jsonl 이름으로 저장
            out_name = f"batchoutput_{idx:04d}.jsonl"
            out_path = output_dir / out_name

            with out_path.open("w", encoding="utf-8") as out_f:
                out_f.write(content)

            print(f"[OK] Saved output to: {out_path}")
            idx += 1

    print("\n[DONE] All done.")


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig) -> None:
    retrieve_batches(cfg)


if __name__ == "__main__":
    main()
