# Data-Augmentation-with-LLMs

## How to run?

no requirements.txt bs. just `uv run`

if some of the generation does not work as expected, it may be because the sdk have not been added on your end locally

For OpenAI: `uv add openai`
For XAI (Grok): `uv add xai_sdk`
For gemini: `uv add google-genai`

### GPT (OpenAI gpt-4o-mini)

1. Make sure you setup `.env` and `gpt/setting.yaml`
   - Add your XAI_API_KEY into the .env file
   - Editting the `setting.yaml` in the `grok` folder accordingly
2. To generate batch files, `uv run python .\gpt\prompt.py`
3. To submit batch files to generate, `uv run --env-file .env python .\gpt\generate.py`
4. To retreive the result of batch, `uv run --env-file .env python .\gpt\retreive.py`
5. To convert jsonl batch output into clean data, `uv run python .\gpt\convert_to_dataset.py`

### Grok (grok-3-mini)

1. Make sure you setup `.env` and `grok/setting.yaml`
   - Add your XAI_API_KEY into the .env file
   - Editting the `setting.yaml` in the `grok` folder accordingly
2. [WORKING] To generate batch files, `uv run python .\grok\grok_prompt.py`
3. To submit batch files to generate, `uv run --env-file .env python .\grok\grok_generate.py`
4. To retreive the result of batch, `uv run --env-file .env python .\grok\grok_retreive.py`
5. To convert jsonl batch output into clean data, `uv run python .\grok\convert_to_dataset.py`

### Deepseek (deepseek-chat)

1. Make sure you setup `.env` and `deepseek/setting.yaml`
   - Add your `DEEPSEEK_API_KEY` into the .env file
   - Editting the `setting.yaml` in the `deepseek` folder accordingly
2. [WORKING] To generate batch files, `uv run python .\deepseek\deepseek_prompt.py`
3. To submit batch files to generate, `uv run --env-file .env python .\deepseek\deepseek_generate.py`
4. To retreive the result of batch, `uv run --env-file .env python .\deepseek\deepseek_retreive.py`
5. To convert jsonl batch output into clean data, `uv run python .\deepseek\deepseek_convert_to_dataset.py`

### Gemini (gemini-2.5-flash-lite)

1. Make sure you setup `.env` and `gemini/setting.yaml`
   - Add your `GEMINI_API_KEY` into the .env file
   - Editting the `setting.yaml` in the `gemini` folder accordingly
2. [WORKING] To generate batch files, `uv run python .\gemini\gem_prompt.py`
3. To submit batch files to generate, `uv run --env-file .env python .\gemini\gem_generate.py`
4. To retreive the result of batch, `uv run --env-file .env python .\gemini\gem_retreive.py`
5. To convert jsonl batch output into clean data, `uv run python .\gemini\gem_convert_to_dataset.py`
