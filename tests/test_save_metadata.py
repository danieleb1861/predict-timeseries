import hashlib
import json
import subprocess

from load_forecasting.save_metadata import build_public_run_summary, collect_run_provenance, save_run_summary


def test_collect_run_provenance_is_complete_and_json_serializable(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    presets = tmp_path / "presets.json"
    specs = tmp_path / "model_specs.json"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    config.write_text('{"config": 1}', encoding="utf-8")
    presets.write_text('{"presets": 1}', encoding="utf-8")
    specs.write_text('{"models": 1}', encoding="utf-8")
    (data_dir / "leg.parquet").write_bytes(b"test parquet placeholder")

    class Result:
        returncode = 0

        def __init__(self, stdout):
            self.stdout = stdout

    outputs = iter((Result("abc123\n"), Result(" M README.md\n")))
    original_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if command[0] == "git":
            return next(outputs)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr("load_forecasting.save_metadata.subprocess.run", fake_run)

    provenance = collect_run_provenance(
        tmp_path,
        config,
        presets,
        specs,
        data_dir,
        ["leg.parquet"],
        "keras",
    )

    assert provenance["git"] == {"commit": "abc123", "dirty": True}
    assert provenance["runtime"]["backend"] == "keras"
    assert provenance["config_sha256"]["system"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert len(provenance["input_fingerprint"]["sha256"]) == 64
    json.dumps(provenance)


def test_collect_run_provenance_tolerates_missing_git(tmp_path, monkeypatch):
    for name in ("config.json", "presets.json", "model_specs.json", "leg.parquet"):
        (tmp_path / name).write_bytes(b"x")

    def missing_git(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("load_forecasting.save_metadata.subprocess.run", missing_git)

    provenance = collect_run_provenance(
        tmp_path,
        tmp_path / "config.json",
        tmp_path / "presets.json",
        tmp_path / "model_specs.json",
        tmp_path,
        ["leg.parquet"],
        "flax",
    )

    assert provenance["git"] == {"commit": None, "dirty": None}


def test_public_run_summary_uses_explicit_allowlist(tmp_path):
    placeholder = "DO_NOT_PUBLISH"
    summary = {
        "created_at": "2026-08-23T00:00:00",
        "preset": "navigation",
        "selector": "navigation",
        "models": ["Ridge"],
        "lookback_steps": 50,
        "horizon_steps": 25,
        "window_sliding_coeff": 5,
        "sampling": {"Ts": "1s"},
        "features": ["act_pow_pu"],
        "targets": ["act_pow_pu"],
        "shapes": {"test_X": [10, 50, 1]},
        "samples": {"test": 10},
        "config_file": f"C:/private/{placeholder}/config.json",
        "preset_file": f"C:/private/{placeholder}/presets.json",
        "model_specs_file": f"C:/private/{placeholder}/models.json",
        "input_files": [f"{placeholder}.parquet"],
        "run_dir": f"C:/private/{placeholder}/run",
        "image_dir": f"C:/private/{placeholder}/img",
        "system_config": {"private_value": placeholder},
        "provenance": {
            "command": ["python", f"C:/private/{placeholder}/train.py"],
            "git": {"commit": "abc123", "dirty": True, "extra": placeholder},
            "runtime": {
                "python": "3.11.9",
                "backend": "keras",
                "devices": ["CPU:0"],
                "extra": placeholder,
            },
            "packages": {"numpy": "2.3.5"},
            "config_sha256": {"system": "a" * 64, "extra": placeholder},
            "input_fingerprint": {
                "method": "sha256(name,size_bytes,mtime_ns)",
                "sha256": "b" * 64,
                "extra": placeholder,
            },
        },
    }

    public = build_public_run_summary(summary)
    serialized = json.dumps(public)

    assert public["public_summary_schema_version"] == 1
    assert public["selector"] == "navigation"
    assert public["provenance"]["git"] == {"commit": "abc123", "dirty": True}
    assert "command" not in public["provenance"]
    assert placeholder not in serialized

    save_run_summary(summary, tmp_path)
    private_saved = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    public_saved = json.loads((tmp_path / "run_summary_public.json").read_text(encoding="utf-8"))
    assert private_saved["system_config"]["private_value"] == placeholder
    assert placeholder not in json.dumps(public_saved)
