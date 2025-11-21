# Data-Augmentation-with-LLMs

## Getting Started

no requirements.txt bs. just `uv run ~~`

### GPT

1. Make sure you setup `.env` and `gpt/setting.yaml`
2. To generate batch files, `uv run python .\gpt\generate.py`
3. To submit batch files to generate, `uv run --env-file .env python .\gpt\generate.py`
4. To retreive the result of batch, `uv run --env-file .env python .\gpt\retreive.py`
5. To convert jsonl batch output into clean data, `uv run python .\gpt\convert_to_dataset.py`

---
### OpenAI
1. Run `test_openAI.py` to test your OpenAI api key is working
---

### Grok

1. Run `test_grok.py` to test your grok api key is working

---

### Some notes (Jae)

#### settings.yaml file

- Batch &rarr; how many request-recorsd are grouped into each input file
- Shots &rarr; how many few-shot examples you include per request. The number of examples taken from the dataset and put into one prompt chunk
- Gen-per-request &rarr; how many new examples you ask the model to generate for each request/record "You will now generate exactly `{gen-per-request}` new examples"

- Suggest to test with
- batch = 10, shots = 2, gen-per-request = 1 first until yall familiarise with it so as to not waste api calls
