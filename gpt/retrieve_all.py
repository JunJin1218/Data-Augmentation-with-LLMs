import subprocess
import datetime
import os

# 실행할 SuperGLUE subset 리스트
SUBSETS = [
    "cb",
    # "rte",
    "copa",
    # "wic",
    "wsc",
    # "boolq", 
    # "multirc",
    # "record"
]

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def run_subset(subset):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{subset}_{timestamp}.log")

    cmd = [
        "uv", "run", "--env-file", ".env", "python", "./gpt/retrieve.py",
        f"subset={subset}"
    ]

    print(f"[RUN] subset={subset}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"[LOG] {log_path}\n")

    with open(log_path, "w", encoding="utf8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            text=True
        )
        process.wait()

    if process.returncode == 0:
        print(f"[OK] {subset} completed\n")
    else:
        print(f"[ERR] {subset} failed with code {process.returncode}\n")


def main():
    for subset in SUBSETS:
        run_subset(subset)


if __name__ == "__main__":
    main()
