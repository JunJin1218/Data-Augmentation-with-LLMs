import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

try:
    # Load .env so XAI_API_KEY can be picked up if defined there
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv is optional; if missing we'll rely on shell env
    pass

from xai_sdk import Client
from xai_sdk.chat import user, system

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from prompts.utils import get_task_name

# Uses XAI_API_KEY from the environment
client = Client()

logger = logging.getLogger("grok_generate")


def _setup_logging(log_dir: str) -> None:
    """
    Configure console + file logging.
    - LOG_LEVEL env var controls verbosity (default: INFO)
    - Writes a grok_generate.log file into log_dir
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger.setLevel(level)

    # Avoid double handlers if Hydra re-invokes main in the same process.
    if logger.handlers:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, "grok_generate.log"),
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # If file logging can't be set up (permissions, etc.), we still keep console logs.
        logger.exception(
            "Failed to set up file logging; continuing with console logging only."
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

    slow_call_s = float(os.getenv("SLOW_CALL_SECONDS", "20"))
    progress_every = int(os.getenv("PROGRESS_EVERY", "25"))

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    t_file_start = time.perf_counter()
    input_name = os.path.basename(input_path)
    output_name = os.path.basename(output_path)

    logger.info("Starting file: %s -> %s", input_name, output_name)

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            t_line_start = time.perf_counter()

            try:
                original = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                logger.warning(
                    "%s line %d: invalid JSON, skipping. (bytes=%d)",
                    input_name,
                    line_no,
                    len(line),
                )
                continue

            meta: Dict[str, Any] = {
                "line_no": line_no,
                "custom_id": original.get("custom_id"),
            }
            custom_id = meta.get("custom_id")

            body = original.get("body", {})
            if not isinstance(body, dict):
                skipped += 1
                logger.warning(
                    "%s line %d (custom_id=%s): missing or invalid 'body', skipping.",
                    input_name,
                    line_no,
                    custom_id,
                )
                continue

            # Prefer the model specified in the batch body, else default
            model = body.get("model") or default_model

            messages = build_messages_from_body(body)
            if not messages:
                skipped += 1
                logger.warning(
                    "%s line %d (custom_id=%s): no 'input' messages found in body, skipping.",
                    input_name,
                    line_no,
                    custom_id,
                )
                continue

            # Log message roles + lengths (avoid logging full prompt contents)
            if logger.isEnabledFor(logging.DEBUG):
                roles = [m.get("role", "user") for m in messages if isinstance(m, dict)]
                lens = [
                    (
                        len(m.get("content", ""))
                        if isinstance(m.get("content", ""), str)
                        else -1
                    )
                    for m in messages
                    if isinstance(m, dict)
                ]
                logger.debug(
                    "%s line %d (custom_id=%s): model=%s messages=%d roles=%s lens=%s",
                    input_name,
                    line_no,
                    custom_id,
                    model,
                    len(messages),
                    roles,
                    lens,
                )

            try:
                t_req_start = time.perf_counter()

                t_create = time.perf_counter()
                chat = client.chat.create(model=model)
                t_create = time.perf_counter() - t_create

                # Reuse the exact system/user messages from the batch
                appended = 0
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
                    appended += 1

                t_sample = time.perf_counter()
                response = chat.sample()
                t_sample = time.perf_counter() - t_sample

                t_req = time.perf_counter() - t_req_start

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "%s line %d (custom_id=%s): timings create=%.3fs sample=%.3fs total=%.3fs appended=%d",
                        input_name,
                        line_no,
                        custom_id,
                        t_create,
                        t_sample,
                        t_req,
                        appended,
                    )

                if t_req >= slow_call_s:
                    logger.warning(
                        "%s line %d (custom_id=%s): SLOW API call (%.2fs >= %.2fs). Possible bottleneck: network/rate limit/model latency.",
                        input_name,
                        line_no,
                        custom_id,
                        t_req,
                        slow_call_s,
                    )

                out_obj = {
                    "input_obj": original,
                    "grok_model": model,
                    "grok_output": response.content,
                    "meta": meta,
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                ok += 1

                dt_line = time.perf_counter() - t_line_start
                logger.info(
                    "%s line %d (custom_id=%s): OK wrote response (%.2fs)",
                    input_name,
                    line_no,
                    custom_id,
                    dt_line,
                )

            except Exception as e:
                failed += 1
                dt_line = time.perf_counter() - t_line_start
                xai_key_set = bool(os.getenv("XAI_API_KEY"))
                logger.error(
                    "%s line %d (custom_id=%s): ERROR Grok request failed after %.2fs | model=%s | XAI_API_KEY_set=%s | err=%s",
                    input_name,
                    line_no,
                    custom_id,
                    dt_line,
                    model,
                    xai_key_set,
                    str(e),
                    exc_info=True,
                )
                continue

            if progress_every > 0 and total % progress_every == 0:
                elapsed = time.perf_counter() - t_file_start
                logger.info(
                    "Progress %s: processed=%d ok=%d skipped=%d failed=%d elapsed=%.1fs",
                    input_name,
                    total,
                    ok,
                    skipped,
                    failed,
                    elapsed,
                )

    elapsed = time.perf_counter() - t_file_start
    logger.info(
        "Finished file: %s | processed=%d ok=%d skipped=%d failed=%d | elapsed=%.1fs | output=%s",
        input_name,
        total,
        ok,
        skipped,
        failed,
        elapsed,
        output_name,
    )


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

    _setup_logging(grok_output_dir)

    logger.info("Task:            %s", task)
    logger.info("Batch input dir: %s", batch_input_dir)
    logger.info("Grok output dir: %s", grok_output_dir)
    logger.info("CWD (Hydra run dir): %s", os.getcwd())

    xai_key_set = bool(os.getenv("XAI_API_KEY"))
    if not xai_key_set:
        logger.warning(
            "XAI_API_KEY is not set in the environment. API calls will likely fail."
        )

    batch_files = get_sorted_batch_files(batch_input_dir)
    if not batch_files:
        logger.warning("No batchinput_*.jsonl files found in: %s", batch_input_dir)

    for filename in batch_files:
        input_path = os.path.join(batch_input_dir, filename)
        # e.g. batchinput_0001.jsonl → grok_output_batchinput_0001.jsonl
        output_filename = f"grok_output_{filename}"
        output_path = os.path.join(grok_output_dir, output_filename)

        logger.info("Processing %s -> %s", filename, output_filename)
        try:
            process_batch_file_with_grok(
                input_path=input_path,
                output_path=output_path,
                default_model="grok-3-mini",
            )
        except Exception as e:
            logger.error(
                "Failed while processing %s: %s", filename, str(e), exc_info=True
            )
            continue


if __name__ == "__main__":
    main()
