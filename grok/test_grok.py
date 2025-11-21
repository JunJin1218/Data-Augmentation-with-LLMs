"""
Simple Grok (xAI) API test script.

Usage (PowerShell):
  $env:XAI_API_KEY = "xai-..."
  python .\gpt\test_grok.py

Or create a `.env` file with:
  XAI_API_KEY=xai-...

This script posts a minimal chat completion request to the xAI endpoint and prints
status code and JSON response to help you confirm your API key and network.
"""

import os
import sys
import json

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import requests

API_KEY = os.environ.get("XAI_API_KEY")
if not API_KEY:
    print("[ERROR] Environment variable XAI_API_KEY not set.")
    print("Set it in PowerShell like: $env:XAI_API_KEY = 'xai-...'")
    sys.exit(1)

URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-3-mini"

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Say only: Hello from Grok!"},
    ],
    "stream": False,
    "temperature": 0,
}

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

print(f"[INFO] Sending request to {URL} with model={MODEL}")
try:
    resp = requests.post(URL, headers=headers, json=payload, timeout=30)
except requests.exceptions.RequestException as e:
    print(f"[ERROR] Request failed: {e}")
    sys.exit(2)

print(f"[INFO] Response status: {resp.status_code}")

# Try to pretty-print JSON response
try:
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
except ValueError:
    print("[WARN] Response was not JSON. Raw body below:\n")
    print(resp.text)

# Quick checks and hints
if resp.status_code == 200:
    print("\n[SUCCESS] API call succeeded. Inspect the JSON above for model output.")
elif resp.status_code == 401:
    print(
        "\n[HINT] 401 Unauthorized — check that XAI_API_KEY is correct and not expired."
    )
elif resp.status_code == 403:
    print(
        "\n[HINT] 403 Forbidden — key may lack permission for this model or endpoint."
    )
elif resp.status_code == 404:
    print("\n[HINT] 404 Not Found — confirm the endpoint URL is correct.")
elif resp.status_code == 429:
    print(
        "\n[HINT] 429 Rate limit — you've hit request limits. Back off or check quotas."
    )
else:
    print("\n[HINT] Check the printed JSON for error details from the server.")
