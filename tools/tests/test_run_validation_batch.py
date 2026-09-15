import json

from run_validation_batch import run_batch
from verify import VerificationResult


def test_run_batch_counts_and_writes_stats(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "business_informatics.txt").write_text("prompt")
    out_dir = tmp_path / "out"

    call_count = {"n": 0}

    def fake_generate_problem(config, template_path, **kwargs):
        call_count["n"] += 1
        return {"id": call_count["n"]}

    def fake_verify(raw):
        # every other problem passes
        accepted = raw["id"] % 2 == 0
        reasons = [] if accepted else ["computed 'saddle' expected 'center'"]
        return VerificationResult(accepted=accepted, reasons=reasons)

    monkeypatch.setattr("run_validation_batch.generate_problem", fake_generate_problem)
    monkeypatch.setattr("run_validation_batch.verify", fake_verify)
    monkeypatch.setattr("run_validation_batch.TRACKS", ["business_informatics"])

    from run_validation_batch import GenerationConfig

    stats = run_batch(
        GenerationConfig(api_key="fake", folder_id="fake"),
        prompts_dir,
        out_dir,
        per_track=4,
    )

    assert stats["generated"] == 4
    assert stats["accepted"] == 2
    assert stats["rejected_auto"] == 2
    assert (out_dir / "validation_stats.json").exists()
    written = json.loads((out_dir / "validation_stats.json").read_text())
    assert written["totals"]["accepted"] == 2
