#!/usr/bin/env python
"""
generate.py (Gemini)

Local "batch runner" for dataset generation.

This script reads JSONL request files produced by prompt.py:
  gpt/batches/{task}/batchinput_*.jsonl

Each line is expected to look similar to an OpenAI batch request, e.g.
{
  "custom_id": "request-1",
  "method": "POST",
  "url": "/v1/responses",
  "body": {
    "model": "gemini-2.5-flash-lite",
    "input": [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
    "text": {... schema ...}
  }
}

It then calls the Gemini API (Google GenAI SDK) *synchronously* and writes outputs to:
  data/{task}/{model}/batchoutput_XXXX.jsonl

The output JSONL is wrapped in an OpenAI-like shape so convert_to_dataset.py can remain unchanged.

Auth:
- Set either GEMINI_API_KEY or GOOGLE_API_KEY as an environment variable.
  (If both are set, GOOGLE_API_KEY typically takes precedence.)
"""

from __future__ import annotations

import json
import time
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Iterable

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from google import genai
from google.genai import types


def _manual_load_env(env_path: Path):
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


# use Hydra original cwd if available, else current file's parent chain
try:
    import hydra

    root = Path(hydra.utils.get_original_cwd())
except Exception:
    root = Path(__file__).resolve().parents[1]

_manual_load_env(root / ".env")


def _load_dotenv_if_present() -> None:
    """Load .env from the *original* working directory (Hydra changes CWD)."""
    try:
        from dotenv import load_dotenv  # type: ignore

        # Use Hydra helper so this works even after Hydra changes the working directory.
        load_dotenv(dotenv_path=to_absolute_path(".env"))
    except Exception:
        # It's fine if python-dotenv isn't installed or .env doesn't exist.
        pass


def _get_api_key() -> str:
    # Support common env var names
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Missing Gemini API key. Set GEMINI_API_KEY (recommended) or GOOGLE_API_KEY, "
            "or put it in a .env file and install python-dotenv."
        )
    return api_key


from prompts.utils import get_task_name


def _as_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s or None
    return str(x).strip() or None


def _extract_system_user_from_messages(
    msgs: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Supports:
    - OpenAI-style list[{"role": "...", "content": "..."}]
    - Plain string
    """
    if isinstance(msgs, str):
        return None, msgs

    if not isinstance(msgs, list):
        return None, _as_str(msgs)

    system: Optional[str] = None
    user: Optional[str] = None
    user_parts: list[str] = []

    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = _as_str(m.get("content"))
        if not content:
            continue
        if role == "system":
            system = content
        elif role == "user":
            user = content
        else:
            # keep any other roles as part of the user prompt (best-effort)
            user_parts.append(f"[{role}] {content}" if role else content)

    if user is None and user_parts:
        user = "\n\n".join(user_parts).strip()

    return system, user


def _maybe_json_load(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _extract_response_json_schema(schema_payload: Any) -> Optional[Dict[str, Any]]:
    """
    Gemini structured outputs expects a JSON Schema dict.
    Your existing schema.json might be:
    - already a JSON Schema dict, OR
    - an OpenAI wrapper that contains a JSON Schema under a nested key.

    This function tries to find the JSON Schema defensively.
    """
    schema_payload = _maybe_json_load(schema_payload)

    if not isinstance(schema_payload, dict):
        return None

    # Direct keys (Gemini-style or generic)
    for k in ("response_json_schema", "json_schema", "schema"):
        v = schema_payload.get(k)
        if isinstance(v, dict):
            return v

    # OpenAI-ish wrappers sometimes include "format" or "response_format"
    fmt = schema_payload.get("format")
    if isinstance(fmt, dict):
        for k in ("response_json_schema", "json_schema", "schema"):
            v = fmt.get(k)
            if isinstance(v, dict):
                return v

    rf = schema_payload.get("response_format")
    if isinstance(rf, dict):
        for k in ("response_json_schema", "json_schema", "schema"):
            v = rf.get(k)
            if isinstance(v, dict):
                return v

    # If the payload itself looks like a JSON Schema, use it as-is.
    if (
        "properties" in schema_payload
        or "$schema" in schema_payload
        or schema_payload.get("type")
        in {
            "object",
            "array",
            "string",
            "number",
            "integer",
            "boolean",
            "null",
        }
    ):
        return schema_payload

    return None


def _wrap_openai_like_output(custom_id: Optional[str], text: str) -> Dict[str, Any]:
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                            }
                        ]
                    }
                ]
            }
        },
    }


def _call_gemini_with_retries(
    client: genai.Client,
    model: str,
    system_instruction: Optional[str],
    user_text: str,
    response_json_schema: Optional[Dict[str, Any]],
    cfg: DictConfig,
) -> str:
    retries = int(cfg.get("retries", 5))
    base_sleep = float(cfg.get("retry_base_sleep_s", 1.0))
    max_sleep = float(cfg.get("retry_max_sleep_s", 30.0))

    temperature = cfg.get("temperature", None)
    max_output_tokens = cfg.get("max_output_tokens", None)
    top_p = cfg.get("top_p", None)
    top_k = cfg.get("top_k", None)

    # Build config
    config_kwargs: Dict[str, Any] = {}

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    # Best-effort structured output in JSON:
    # Use both mime + schema when we have a schema, otherwise mime only.
    config_kwargs["response_mime_type"] = "application/json"
    if response_json_schema:
        config_kwargs["response_json_schema"] = response_json_schema

    # Optional sampling/length controls
    if temperature is not None:
        config_kwargs["temperature"] = float(temperature)
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = int(max_output_tokens)
    if top_p is not None:
        config_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        config_kwargs["top_k"] = int(top_k)

    gen_config = types.GenerateContentConfig(**config_kwargs)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_text,
                config=gen_config,
            )
            return resp.text or ""
        except Exception as e:
            last_err = e
            if attempt == retries - 1:
                break
            sleep_s = min(max_sleep, base_sleep * (2**attempt))
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Gemini call failed after {retries} attempts: {last_err}"
    ) from last_err


def run(cfg: DictConfig) -> None:
    task = get_task_name(cfg)
    model_name = cfg.model  # e.g. "gemini-2.5-flash-lite"

    # Hydra changes CWD; make these absolute.
    input_dir = Path(to_absolute_path(str(Path("gpt") / "batches" / task)))
    output_dir = Path(to_absolute_path(str(Path("data") / task / model_name)))
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(input_dir.glob("batchinput_*.jsonl"))
    if not batch_files:
        raise FileNotFoundError(f"No batchinput_*.jsonl found under: {input_dir}")

    # Create Gemini client
    _load_dotenv_if_present()
    client = genai.Client(api_key=_get_api_key())

    print(f"[INFO] Task          : {task}")
    print(f"[INFO] Model         : {model_name}")
    print(f"[INFO] Input dir     : {input_dir}")
    print(f"[INFO] Output dir    : {output_dir}")
    print(f"[INFO] Num input files: {len(batch_files)}")

    per_request_sleep = float(cfg.get("sleep_s", 0.0))
    overwrite = bool(cfg.get("overwrite", False))

    for file_idx, in_path in enumerate(batch_files, start=1):
        out_name = f"batchoutput_{file_idx:04d}.jsonl"
        out_path = output_dir / out_name

        if out_path.exists() and not overwrite:
            print(
                f"[SKIP] {out_name} already exists (set overwrite=true to regenerate)."
            )
            continue

        print(f"\n[INFO] Processing {in_path.name} -> {out_name}")

        with (
            in_path.open("r", encoding="utf-8") as f_in,
            out_path.open("w", encoding="utf-8") as f_out,
        ):
            for line_idx, line in enumerate(tqdm(f_in, desc=in_path.name), start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    f_out.write(
                        json.dumps(
                            {"line_idx": line_idx, "error": "invalid_json"},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                custom_id = rec.get("custom_id")
                body = rec.get("body", rec)

                # Body fields from prompt.py (OpenAI-style)
                req_model = body.get("model") or model_name
                msgs = body.get("input") or body.get("contents") or body.get("prompt")
                schema_payload = body.get("text") or body.get("schema")

                system_instruction, user_text = _extract_system_user_from_messages(msgs)
                user_text = user_text or ""
                response_schema = _extract_response_json_schema(schema_payload)

                try:
                    text = _call_gemini_with_retries(
                        client=client,
                        model=req_model,
                        system_instruction=system_instruction,
                        user_text=user_text,
                        response_json_schema=response_schema,
                        cfg=cfg,
                    )
                    out_entry = _wrap_openai_like_output(custom_id, text)
                except Exception as e:
                    out_entry = {
                        "custom_id": custom_id,
                        "error": {
                            "type": e.__class__.__name__,
                            "message": str(e),
                        },
                    }

                f_out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")

                if per_request_sleep > 0:
                    time.sleep(per_request_sleep)

        print(f"[OK] Saved: {out_path}")


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
