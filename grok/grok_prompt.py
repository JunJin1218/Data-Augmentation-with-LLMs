import os
import json
import logging
from tqdm import tqdm
from datasets import load_dataset

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from prompts.utils import get_user_prompt_fn, get_task_name

logger = logging.getLogger("grok_prompt")


def load_data(cfg: DictConfig):
    """Load dataset from Hugging Face Hub, optionally shuffled."""
    if hasattr(cfg, "subset") and cfg.subset:
        ds = load_dataset(cfg.dataset, cfg.subset, split="train")
    else:
        ds = load_dataset(cfg.dataset, split="train")

    # Optional shuffling to diversify few-shot chunks
    shuffle_enabled = bool(cfg.get("shuffle", False))
    seed = int(cfg.get("seed", 42))
    if shuffle_enabled:
        try:
            ds = ds.shuffle(seed=seed)
        except Exception:
            logger.warning("Dataset shuffle failed; proceeding without shuffling.")
    return ds

def load_prompts(cfg: DictConfig):
    task = get_task_name(cfg)
    prompt_dir = to_absolute_path(os.path.join("grok", "prompts", task))
    system_path = os.path.join(prompt_dir, "system.txt")
    user_path = os.path.join(prompt_dir, "user.txt")

    with open(system_path, "r", encoding="utf-8") as f:
        system_tmpl = f.read()
    with open(user_path, "r", encoding="utf-8") as f:
        user_tmpl = f.read()

    return system_tmpl, user_tmpl

def load_schema(cfg: DictConfig):
    task = get_task_name(cfg)
    prompt_dir = to_absolute_path(os.path.join("grok", "prompts", task))
    schema_path = os.path.join(prompt_dir, "schema.json")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    return schema

@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig):
    """
        Example setting.yaml:
            model: grok-3-mini
      dataset: glue
      subset: mrpc
      batch: 2000
      shots: 5
    """
    model_name = cfg.model
    batch_size = cfg.batch
    shots = cfg.shots
    task = get_task_name(cfg)
    output_dir = f"grok/batches/{task}"

    # Convert to absolute path because Hydra changes the working directory
    output_dir = to_absolute_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Basic console logging setup once
    if not logger.handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    logger.info("Task: %s", task)
    logger.info("Dataset: %s | Subset: %s", getattr(cfg, "dataset", None), getattr(cfg, "subset", None))
    logger.info("Model: %s | Shots: %s | Gen/req: %s | Batch size: %s", model_name, shots, cfg.get("gen_per_request", 5), batch_size)
    logger.info("Shuffle: %s | Seed: %s", bool(cfg.get("shuffle", False)), int(cfg.get("seed", 42)))
    logger.info("Output dir (batches): %s", output_dir)

    ds = load_data(cfg)
    system_tmpl, user_tmpl = load_prompts(cfg)
    schema = load_schema(cfg)

    records = []
    file_idx = 1
    num_samples = len(ds)

    prompt_fn = get_user_prompt_fn(cfg)
    gen_per_request = cfg.get("gen_per_request", 5)

    # Few-shot sampling strategy for diversity
    sampling_mode = str(cfg.get("fewshot_sampling", "sequential")).lower()  # sequential | global_random
    seed = int(cfg.get("seed", 42))

    if sampling_mode == "global_random":
        # Build a shuffled index over the full dataset, then sample shots per request
        import random
        rng = random.Random(seed)
        indices = list(range(num_samples))
        rng.shuffle(indices)
        # Number of requests we will form (ceil)
        from math import ceil
        num_requests = ceil(num_samples / shots)
        # Distribute indices roughly evenly across requests in a round-robin manner
        buckets = [[] for _ in range(num_requests)]
        for i, idx in enumerate(indices):
            buckets[i % num_requests].append(idx)
        # For each bucket, sample up to `shots` diverse examples
        request_chunks = []
        for b in buckets:
            if not b:
                continue
            rng.shuffle(b)
            sel = b[:shots]
            request_chunks.append(ds.select(sel))
    else:
        # Default: contiguous slices (original behavior)
        request_chunks = []
        for start in range(0, num_samples, shots):
            end = min(start + shots, num_samples)
            request_chunks.append(ds.select(range(start, end)))

    total_requests = len(request_chunks)
    logger.info("Computed requests: %s (approx ceil(train_size/shots))", total_requests)

    for req_idx, chunk in tqdm(enumerate(request_chunks, start=1), total=total_requests, desc=f"Processing {task}"):
        system_prompt = system_tmpl
        user_parts = [user_tmpl.strip(), ""]
        user_parts.append(prompt_fn(chunk))
        user_parts.append(
            f"You will now generate exactly {gen_per_request} new examples. Ensure topical diversity; avoid reusing the same entities/questions."
        )
        user_prompt = "\n".join(user_parts).strip()

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        record = {
            "custom_id": f"request-{req_idx}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model_name,
                "input": prompt,
                "text": schema,
            },
        }
        records.append(record)

        # Write a batch file whenever we reach batch_size or at the final request
        if len(records) >= batch_size or req_idx == total_requests:
            fname = f"batchinput_{file_idx:04d}.jsonl"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info("Saved %s with %d records (task=%s)", fname, len(records), task)
            records = []
            file_idx += 1

if __name__ == "__main__":
    main()
