import argparse
import json
from pathlib import Path

from datasets import load_dataset


def make_task_name(dataset: str, subset: str | None) -> str:
    return f"{dataset}-{subset}" if subset else dataset


def map_dataset_id(dataset: str) -> tuple[str, bool]:
    """Map friendly names to HF repo IDs and whether trust_remote_code is needed."""
    if dataset == "legalbench":
        return "nguha/legalbench", True
    if dataset == "biosses":
        return "tabilab/biosses", False
    return dataset, False


def save_jsonl(ds, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create train/test split JSONL files.")
    parser.add_argument("--dataset", required=True, help="Dataset name or alias (e.g., biosses, legalbench, super_glue)")
    parser.add_argument("--subset", default=None, help="Optional subset/config (e.g., legal_reasoning_causality, cb)")
    parser.add_argument("--test-pct", type=float, default=0.2, help="Fraction for test split, e.g., 0.2 for 20%")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    dataset_id, trust_remote_code = map_dataset_id(args.dataset)

    # Prefer train split if available; otherwise fall back to full default split
    split = "train"
    try:
        ds = load_dataset(dataset_id, args.subset, split=split, trust_remote_code=trust_remote_code)
    except Exception:
        ds = load_dataset(dataset_id, args.subset, trust_remote_code=trust_remote_code)[split] if split in load_dataset(dataset_id, args.subset, trust_remote_code=trust_remote_code) else load_dataset(dataset_id, args.subset, trust_remote_code=trust_remote_code)[0]

    parts = ds.train_test_split(test_size=args.test_pct, seed=args.seed)
    train_ds = parts["train"]
    test_ds = parts["test"]

    task = make_task_name(args.dataset, args.subset)
    base_dir = Path("data") / "splits" / task
    save_jsonl(train_ds, base_dir / "train.jsonl")
    save_jsonl(test_ds, base_dir / "test.jsonl")

    print(f"[INFO] Wrote train/test to {base_dir}")


if __name__ == "__main__":
    main()
