import os
import json
from tqdm import tqdm
from datasets import load_dataset

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from prompts.utils import get_user_prompt_fn, get_task_name


def load_data(cfg: DictConfig):
    """Load dataset from Hugging Face Hub."""
    dataset_name = cfg.dataset
    trust_remote_code = False

    # Handle LegalBench mapping
    if dataset_name == "legalbench":
        dataset_name = "nguha/legalbench"
        trust_remote_code = True

    if dataset_name == "biosses":
        from pathlib import Path
        from hydra.utils import to_absolute_path
        local_train = Path(to_absolute_path("data/splits/biosses/train.jsonl"))
        if local_train.exists():
            ds = load_dataset("json", data_files=str(local_train))
            ds = ds["train"] if "train" in ds else ds
            print("BIOSSES")
            return ds
        else:
            raise ValueError("씨발")

    if hasattr(cfg, "subset") and cfg.subset:
        ds = load_dataset(dataset_name, cfg.subset, split="train", trust_remote_code=trust_remote_code)
    else:
        ds = load_dataset(dataset_name, split="train", trust_remote_code=trust_remote_code)
    return ds

def load_prompts(cfg: DictConfig):
    if (cfg.dataset not in ["biosses"]):
        task = get_task_name(cfg)
    else:
        task = cfg.dataset

    prompt_dir = to_absolute_path(os.path.join("gpt", "prompts", task))
    system_path = os.path.join(prompt_dir, "system.txt")
    user_path = os.path.join(prompt_dir, "user.txt")

    with open(system_path, "r", encoding="utf-8") as f:
        system_tmpl = f.read()
    with open(user_path, "r", encoding="utf-8") as f:
        user_tmpl = f.read()

    return system_tmpl, user_tmpl

def load_schema(cfg: DictConfig):
    if (cfg.dataset not in ["biosses"]):
        task = get_task_name(cfg)
    else:
        task = cfg.dataset

    prompt_dir = to_absolute_path(os.path.join("gpt", "prompts", task))
    schema_path = os.path.join(prompt_dir, "schema.json")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    return schema

@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig):
    """
    Example setting.yaml:
      model: gpt-4o-mini
      dataset: glue
      subset: mrpc
      batch: 2000
      shots: 5
    """
    model_name = cfg.model
    batch_size = cfg.batch
    shots = cfg.shots
    if (cfg.dataset not in ["biosses"]):
        task = get_task_name(cfg)
    else:
        task = cfg.dataset
    output_dir = f"gpt/batches/{task}"

    # Convert to absolute path because Hydra changes the working directory
    output_dir = to_absolute_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ds = load_data(cfg)
    ds.shuffle(seed=cfg.seed) # data shuffle
    system_tmpl, user_tmpl = load_prompts(cfg)
    schema = load_schema(cfg)

    records = []
    file_idx = 1
    num_samples = len(ds)

    prompt_fn = get_user_prompt_fn(cfg)
    gen_per_request = cfg.get("gen_per_request", 5)
    

    for start in tqdm(range(0, num_samples, shots), desc="Processing samples"):
        end = min(start + shots, num_samples)
        chunk = ds.select(range(start, end))

        system_prompt = system_tmpl
        user_parts = [user_tmpl.strip(), ""]
        user_parts.append(prompt_fn(chunk))
        user_parts.append(f"You will now generate exactly {gen_per_request} new examples.")
        user_prompt = "\n".join(user_parts).strip()

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        record = {
            "custom_id": f"request-{start//shots+1}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_name,
                "input": prompt,
                "text": schema,
            },
        }
        records.append(record)

        if len(records) >= batch_size or end == num_samples:
            fname = f"batchinput_{file_idx:04d}.jsonl"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Saved {fname} with {len(records)} records.")
            records = []
            file_idx += 1

if __name__ == "__main__":
    main()
