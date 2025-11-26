import os
import json
from typing import Any, Dict, List, Optional

from xai_sdk import Client
from xai_sdk.chat import user, system

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from prompts.utils import get_task_name

# Uses XAI_API_KEY from the environment
client = Client()


def get_sorted_batch_files(batch_input_dir: str) -> List[str]:
    """
    Find all batchinput_*.jsonl files in a directory and return them sorted.
    """
    files = [
        f
        for f in os.listdir(batch_input_dir)
        if f.startswith("batchinput_") and f.endswith(".jsonl")
    ]
    return sorted(files)


def build_messages_from_body(body: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Given the 'body' field from a batch line, return the list of chat messages
    we should send to Grok.

    grok_prompt.py writes lines like:

        {
          "custom_id": "...",
          "method": "POST",
          "url": "/v1/responses",
          "body": {
            "model": "grok-3-mini",
            "input": [
              {"role": "system", "content": "..."},
              {"role": "user",   "content": "..."}
            ],
            "text": { ... JSON schema ... }
          }
        }

    Here we just reuse body["input"] as-is.
    """
    inp = body.get("input")
    if isinstance(inp, list):
        return [m for m in inp if isinstance(m, dict) and "content" in m]
    return None


def process_batch_file_with_grok(
    input_path: str,
    output_path: str,
    default_model: str = "grok-3-mini",
):
    """
    Read a batchinput_*.jsonl file line by line, send each entry to Grok 3 Mini,
    and write results to an output JSONL file.

    Each output line:

        {
          "input_obj": <original JSON object>,
          "grok_model": "grok-3-mini",
          "grok_output": "<string>",
          "meta": {...}
        }
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                original = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[WARN] {os.path.basename(input_path)} line {line_no}: "
                    f"invalid JSON, skipping."
                )
                continue

            meta: Dict[str, Any] = {
                "line_no": line_no,
                "custom_id": original.get("custom_id"),
            }

            body = original.get("body", {})
            if not isinstance(body, dict):
                print(
                    f"[WARN] {os.path.basename(input_path)} line {line_no}: "
                    f"missing or invalid 'body', skipping."
                )
                continue

            # Prefer the model specified in the batch body, else default
            model = body.get("model") or default_model

            messages = build_messages_from_body(body)
            if not messages:
                print(
                    f"[WARN] {os.path.basename(input_path)} line {line_no}: "
                    f"no 'input' messages found in body, skipping."
                )
                continue

            try:
                chat = client.chat.create(model=model)

                # Reuse the exact system/user messages from the batch
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if not isinstance(content, str):
                        continue
                    if role == "system":
                        chat.append(system(content))
                    else:
                        # treat user/assistant/other as user messages
                        chat.append(user(content))

                response = chat.sample()

                out_obj = {
                    "input_obj": original,
                    "grok_model": model,
                    "grok_output": response.content,
                    "meta": meta,
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                print(
                    f"[OK] {os.path.basename(input_path)} line {line_no} "
                    f"→ wrote response"
                )
            except Exception as e:
                print(
                    f"[ERROR] Grok request failed for "
                    f"{os.path.basename(input_path)} line {line_no}: {e}"
                )
                continue


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig):
    """
    Hydra entrypoint.

    Behavior:

    - Determine task name via get_task_name(cfg)
    - Read batchinput_*.jsonl from grok/batches/{task} (created by grok_prompt.py)
    - For each batch file, call Grok 3 Mini using the stored messages
    - Write outputs to grok/outputs/{task}/grok_output_batchinput_XXXX.jsonl
    """
    task = get_task_name(cfg)

    # IMPORTANT: use Grok batches, not GPT batches
    default_input_dir = os.path.join("grok", "batches", task)
    batch_input_dir = cfg.get("output_dir", default_input_dir)
    batch_input_dir = to_absolute_path(batch_input_dir)

    # Where we'll write Grok outputs
    default_output_dir = os.path.join("grok", "outputs", task)
    grok_output_dir = cfg.get("grok_output_dir", default_output_dir)
    grok_output_dir = to_absolute_path(grok_output_dir)

    os.makedirs(grok_output_dir, exist_ok=True)

    print(f"[INFO] Task:            {task}")
    print(f"[INFO] Batch input dir: {batch_input_dir}")
    print(f"[INFO] Grok output dir: {grok_output_dir}")

    for filename in get_sorted_batch_files(batch_input_dir):
        input_path = os.path.join(batch_input_dir, filename)
        # e.g. batchinput_0001.jsonl → grok_output_batchinput_0001.jsonl
        output_filename = f"grok_output_{filename}"
        output_path = os.path.join(grok_output_dir, output_filename)

        print(f"[INFO] Processing {filename} → {output_filename}")
        try:
            process_batch_file_with_grok(
                input_path=input_path,
                output_path=output_path,
                default_model="grok-3-mini",
            )
        except Exception as e:
            print(f"[ERROR] Failed while processing {filename}: {e}")
            continue


if __name__ == "__main__":
    main()
