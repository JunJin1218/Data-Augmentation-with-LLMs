#!/usr/bin/env python3
import random
from pathlib import Path

SAMPLE_K = 10  # 폴더마다 뽑을 개수

def sample_lines(lines, k, rng):
    if len(lines) <= k:
        rng.shuffle(lines)
        return lines
    return rng.sample(lines, k)

def main():
    # 이 파일 기준으로 batches 디렉터리
    root = Path(__file__).resolve().parent
    rng = random.Random()  # 필요시 rng.seed(42)

    if not root.is_dir():
        raise SystemExit(f"[ERR] Not found: {root}")

    # batches 하위의 1뎁스 디렉터리들을 순회 (super_glue-*)
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        in_path = task_dir / "batchinput_0001.jsonl"
        out_path = task_dir / "batch_test.jsonl"

        if not in_path.is_file():
            print(f"[SKIP] {task_dir.name}: no batchinput_0001.jsonl")
            continue

        # 읽고 샘플링
        lines = [ln.rstrip("\n") for ln in in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            print(f"[SKIP] {task_dir.name}: empty input")
            continue

        picked = sample_lines(lines, SAMPLE_K, rng)

        # 쓰기 (덮어쓰기)
        out_path.write_text("\n".join(picked) + "\n", encoding="utf-8")
        print(f"[OK] {task_dir.name}: wrote {len(picked)} -> {out_path.name}")

if __name__ == "__main__":
    main()
