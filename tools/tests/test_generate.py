import json

import pytest

from generate import GenerationConfig, generate_problem, render_prompt


def test_render_prompt_substitutes_variables(tmp_path):
    template = tmp_path / "t.txt"
    template.write_text("Лабораторная №{lab_number}, направление {track}.")
    result = render_prompt(str(template), lab_number=2, track="infosec")
    assert result == "Лабораторная №2, направление infosec."


def test_generate_problem_parses_llm_json(tmp_path, monkeypatch):
    template = tmp_path / "t.txt"
    template.write_text("prompt for {track}")

    def fake_call_llm(config, prompt):
        assert "infosec" in prompt
        return json.dumps({"track": "infosec", "ok": True})

    monkeypatch.setattr("generate.call_llm", fake_call_llm)
    config = GenerationConfig(api_key="fake", folder_id="fake-folder")
    result = generate_problem(config, str(template), track="infosec")
    assert result == {"track": "infosec", "ok": True}


def test_generate_problem_raises_on_malformed_json(tmp_path, monkeypatch):
    template = tmp_path / "t.txt"
    template.write_text("prompt")
    monkeypatch.setattr("generate.call_llm", lambda config, prompt: "not json")
    config = GenerationConfig(api_key="fake", folder_id="fake-folder")
    with pytest.raises(json.JSONDecodeError):
        generate_problem(config, str(template))
