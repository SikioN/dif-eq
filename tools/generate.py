import json
from dataclasses import dataclass

import requests


@dataclass
class GenerationConfig:
    api_key: str
    folder_id: str
    model: str = "yandexgpt/latest"
    endpoint: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    temperature: float = 0.3
    max_tokens: int = 2000


def render_prompt(template_path: str, **template_vars) -> str:
    with open(template_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    return template.format(**template_vars)


def call_llm(config: GenerationConfig, prompt: str) -> str:
    response = requests.post(
        config.endpoint,
        headers={"Authorization": f"Api-Key {config.api_key}"},
        json={
            "modelUri": f"gpt://{config.folder_id}/{config.model}",
            "completionOptions": {
                "temperature": config.temperature,
                "maxTokens": config.max_tokens,
            },
            "messages": [{"role": "user", "text": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["result"]["alternatives"][0]["message"]["text"]


def generate_problem(config: GenerationConfig, template_path: str, **template_vars) -> dict:
    prompt = render_prompt(template_path, **template_vars)
    raw_text = call_llm(config, prompt)
    return json.loads(raw_text)
