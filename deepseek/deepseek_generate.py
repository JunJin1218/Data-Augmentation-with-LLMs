import os
import json
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from prompts.utils import get_task_name


# Configure OpenAI SDK to talk to DeepSeek instead of OpenAI.
# Uses DEEPSEEK_API_KEY from environment and DeepSeek's base URL.
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


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


def extract_messages_from_record(obj: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Given one JSONL record produced by deepseek_prompt.py (OpenAI-batch style),
    extract the chat 'messages' we want to send to DeepSeek.

    Expected shape (from deepseek_prompt.py):

        {
          "custom_id": "...",
          "method": "POST",
          "url": "/v1/responses",
          "body": {
            "model": "deepseek-chat",
            "input": [
              {"role": "system", "content": "..."},
              {"role": "user", "content": "..."}
            ],
            "text": { ... schema.json ... }
          }
        }

    We will:
      - Use body["input"] as messages if it's a list
      - Fall back to wrapping a string in a single user message if needed
    """
    meta: Dict[str, Any] = {"custom_id": obj.get("custom_id")}

    body = obj.get("body", {})
    messages = body.get("input")
    if isinstance(messages, list) and messages:
        # assume already in OpenAI/DeepSeek chat format
        return messages, meta

    # Fallbacks: treat 'input' or 'prompt' as raw text
    raw = body.get("input") or obj.get("input") or obj.get("prompt")
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}], meta

    raise ValueError("Could not extract messages from record; check JSONL schema.")


def process_batch_file_with_deepseek(
    input_path: str,
    output_path: str,
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> None:
    """
    Read a batchinput_*.jsonl file, send each record to DeepSeek via chat.completions,
    and write results to output_path as JSONL.

    Each output line looks like:

        {
          "input_obj": <original JSON object or raw line>,
          "deepseek_model": "deepseek-chat",
          "deepseek_output": "<string>",
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

            original: Any = None
            messages: Optional[List[Dict[str, str]]] = None
            meta: Dict[str, Any] = {"line_no": line_no}

            try:
                original = json.loads(line)
                msgs, extra_meta = extract_messages_from_record(original)
                messages = msgs
                meta.update(extra_meta)
            except json.JSONDecodeError:
                # Plain text line: treat as a single user message
                original = line
                messages = [{"role": "user", "content": line}]
                meta["source"] = "raw_line"
            except ValueError as e:
                print(
                    f"[WARN] {os.path.basename(input_path)} line {line_no}: "
                    f"{e} Skipping."
                )
                continue

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                )
                # DeepSeek is OpenAI-compatible → use choices[0].message.content
                content = None
                if response.choices:
                    content = response.choices[0].message.content

                if not isinstance(content, str):
                    print(
                        f"[WARN] {os.path.basename(input_path)} line {line_no}: "
                        f"no text content in response, skipping."
                    )
                    continue

                out_obj = {
                    "input_obj": original,
                    "deepseek_model": model,
                    "deepseek_output": content,
                    "meta": meta,
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                print(
                    f"[OK] {os.path.basename(input_path)} line {line_no} "
                    f"→ wrote response"
                )
            except Exception as e:
                print(
                    f"[ERROR] DeepSeek request failed for "
                    f"{os.path.basename(input_path)} line {line_no}: {e}"
                )
                # continue with next line
                continue


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint for DeepSeek generation.

    Behavior:

      - Determine task name via get_task_name(cfg)
      - Determine where batchinput_*.jsonl files live
      - For each batchinput file, call DeepSeek chat.completions on each line
      - Save outputs to deepseek/outputs/{task}/deepseek_output_*.jsonl

    Config knobs (optional):

      - cfg.model: DeepSeek model name (e.g., "deepseek-chat" or "deepseek-reasoner")
      - cfg.batch_input_dir: override input dir
      - cfg.output_dir: (fallback) where prompt/batch files are; default gpt/batches/{task}
      - cfg.deepseek_output_dir: where to store DeepSeek outputs
      - cfg.temperature, cfg.top_p: sampling params
    """
    task = get_task_name(cfg)
    model_name = cfg.model

    # Where the batchinput_*.jsonl files live
    default_input_dir = os.path.join("gpt", "batches", task)
    batch_input_dir = cfg.get("batch_input_dir", cfg.get("output_dir", default_input_dir))
    batch_input_dir = to_absolute_path(batch_input_dir)

    # Where we'll write DeepSeek outputs
    default_output_dir = os.path.join("deepseek", "outputs", task)
    deepseek_output_dir = cfg.get("deepseek_output_dir", default_output_dir)
    deepseek_output_dir = to_absolute_path(deepseek_output_dir)
    os.makedirs(deepseek_output_dir, exist_ok=True)

    temperature = float(cfg.get("temperature", 0.7))
    top_p = float(cfg.get("top_p", 0.95))

    print(f"[INFO] Task            : {task}")
    print(f"[INFO] Model           : {model_name}")
    print(f"[INFO] Batch input dir : {batch_input_dir}")
    print(f"[INFO] DeepSeek out dir: {deepseek_output_dir}")
    print(f"[INFO] temperature     : {temperature}")
    print(f"[INFO] top_p           : {top_p}")

    batch_files = get_sorted_batch_files(batch_input_dir)
    if not batch_files:
        raise FileNotFoundError(
            f"No batchinput_*.jsonl files found in {batch_input_dir}. "
            f"Run deepseek_prompt.py first."
        )

    for filename in batch_files:
        input_path = os.path.join(batch_input_dir, filename)
        output_filename = f"deepseek_output_{filename}"
        output_path = os.path.join(deepseek_output_dir, output_filename)

        print(f"[INFO] Processing {filename} → {output_filename}")
        try:
            process_batch_file_with_deepseek(
                input_path=input_path,
                output_path=output_path,
                model=model_name,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            print(f"[ERROR] Failed while processing {filename}: {e}")
            # continue to next file
            continue


if __name__ == "__main__":
    main()
