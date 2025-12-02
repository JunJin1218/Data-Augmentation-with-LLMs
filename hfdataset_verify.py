from datasets import load_dataset
from pprint import pprint

## THIS PY is to just see the structure of superglue tasks' HF dataset

# (dataset_name, subset_name, split_name)
TASKS = [
    # SuperGLUE
    ("super_glue", "cb", "train"),
    ("super_glue", "rte", "train"),
    ("super_glue", "copa", "train"),
    ("super_glue", "wic", "train"),
    ("super_glue", "wsc", "train"),
    ("super_glue", "boolq", "train"),
    ("super_glue", "multirc", "train"),
    ("super_glue", "record", "train"),
]


def inspect_split(dataset_name: str, subset_name: str, split: str = "train", num_examples: int = 2):
    print("=" * 80)
    print(f"DATASET: {dataset_name} | SUBSET: {subset_name} | SPLIT: {split}")
    print("-" * 80)

    ds = load_dataset(dataset_name, subset_name)
    split_ds = ds[split]

    # 1) 전체 피처(컬럼) 구조 보기
    print("[features]")
    pprint(split_ds.features)
    print()

    # 2) 길이
    print(f"[num_examples] {len(split_ds)}\n")

    # 3) 앞부분 몇 개 raw 예시 출력
    print(f"[first {num_examples} examples]")
    for i in range(num_examples):
        print(f"\n--- example {i} ---")
        pprint(split_ds[i])
    print("\n")


def main():
    for dataset_name, subset_name, split in TASKS:
        inspect_split(dataset_name, subset_name, split, num_examples=1)


if __name__ == "__main__":
    main()
