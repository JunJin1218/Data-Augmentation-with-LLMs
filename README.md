# Data-Augmentation-with-LLMs

## How to run?

no requirements.txt bs. just `uv run ~~`

### GPT

1. Make sure you setup `.env` and `gpt/setting.yaml`
   - Add your XAI_API_KEY into the .env file
   - Editting the `setting.yaml` in the `grok` folder accordingly
2. To generate batch files, `uv run python .\gpt\prompt.py`
3. To submit batch files to generate, `uv run --env-file .env python .\gpt\generate.py`
4. To retreive the result of batch, `uv run --env-file .env python .\gpt\retreive.py`
5. To convert jsonl batch output into clean data, `uv run python .\gpt\convert_to_dataset.py`

### Grok

1. Make sure you setup `.env` and `gpt/setting.yaml`
   - Add your XAI_API_KEY into the .env file
   - Editting the `setting.yaml` in the `grok` folder accordingly
2. [WORKING] To generate batch files, `uv run python .\grok\grok_prompt.py`
3. To submit batch files to generate, `uv run --env-file .env python .\grok\grok_generate.py`
4. To retreive the result of batch, `uv run --env-file .env python .\grok\grok_retreive.py`
5. To convert jsonl batch output into clean data, `uv run python .\grok\convert_to_dataset.py`
