import os
import json
from openai import OpenAI

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from prompts.utils import get_task_name

client = OpenAI()


def get_sorted_batch_files(batch_input_dir: str):
    files = [
        f
        for f in os.listdir(batch_input_dir)
        if f.startswith("batchinput_") and f.endswith(".jsonl")
    ]
    return sorted(files)


def record_batch_id(batch_id_list_path: str, filename: str, batch_id: str):
    with open(batch_id_list_path, "a", encoding="utf-8") as f:
        json.dump({filename: batch_id}, f)
        f.write("\n")


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig):
    task = get_task_name(cfg)

    # Convert to absolute path because Hydra changes the working directory
    batch_input_dir = cfg.get("output_dir", os.path.join("gpt", "batchs", task))
    batch_input_dir = to_absolute_path(batch_input_dir)

    batch_id_list_path = to_absolute_path(
        os.path.join("gpt", "batchs", task, f"batch_id_list.jsonl")  # gpt/batchs/{task}/...
    )
    os.makedirs(os.path.dirname(batch_id_list_path), exist_ok=True)

    for filename in get_sorted_batch_files(batch_input_dir):
        filepath = os.path.join(batch_input_dir, filename)

        try:
            with open(filepath, "rb") as f:
                batch_input_file = client.files.create(file=f, purpose="batch")
            print(batch_input_file)

            batch = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/responses",
                completion_window="24h",
            )
            print(batch)
            print()
            print(f"REMEMBER BATCH ID!!!! {batch.id}")

            record_batch_id(batch_id_list_path, filename, batch.id)

        except Exception as e:
            print(f"[ERROR] Failed on {filename}: {e}")
            break


if __name__ == "__main__":
    main()
