"""Generate deterministic synthetic shipboard load data for public demos.

The generated signals are independently simulated and are not fitted to or
sampled from the repository's restricted operational datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "demo_config.json"
DEFAULT_ROWS = {"navigation": 172_800, "manoeuvre": 86_400}
OUTPUT_NAMES = {
    "navigation": "demo_navigation.parquet",
    "manoeuvre": "demo_manoeuvre.parquet",
}


def _load_rating_kw(config_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    rating_kw = float(config["physical_system"]["rating_kW"])
    if rating_kw <= 0.0:
        raise ValueError("physical_system.rating_kW must be positive")
    return rating_kw


def _ar1(num_rows, phi, innovation_std, rng):
    values = np.empty(num_rows, dtype=np.float64)
    values[0] = rng.normal(0.0, innovation_std / np.sqrt(1.0 - phi**2))
    innovations = rng.normal(0.0, innovation_std, size=num_rows - 1)
    for idx in range(1, num_rows):
        values[idx] = phi * values[idx - 1] + innovations[idx - 1]
    return values


def _operating_targets(num_rows, mode, rng):
    if mode == "navigation":
        targets = np.array([
            [0.45, 0.46],
            [0.65, 0.66],
            [0.82, 0.80],
            [0.30, 0.31],
        ])
        weights = np.array([0.25, 0.50, 0.15, 0.10])
        duration_bounds = (600, 3_600)
    else:
        targets = np.array([
            [0.00, 0.00],
            [0.65, 0.65],
            [-0.40, -0.40],
            [-0.25, 0.55],
            [0.55, -0.25],
        ])
        weights = np.array([0.20, 0.35, 0.15, 0.15, 0.15])
        duration_bounds = (20, 240)

    states = np.empty(num_rows, dtype=np.int8)
    start = 0
    while start < num_rows:
        state = int(rng.choice(len(targets), p=weights))
        duration = int(rng.integers(duration_bounds[0], duration_bounds[1] + 1))
        stop = min(start + duration, num_rows)
        states[start:stop] = state
        start = stop
    return states, targets[states]


def _track_command(target, response, noise_std, rng):
    command = np.empty_like(target, dtype=np.float64)
    command[0] = target[0]
    noise = rng.normal(0.0, noise_std, size=len(target) - 1)
    for idx in range(1, len(target)):
        command[idx] = command[idx - 1] + response * (target[idx] - command[idx - 1]) + noise[idx - 1]
    return np.clip(command, -1.0, 1.0)


def _shot_noise(num_rows, probability, magnitude, decay, rng):
    impulses = np.where(
        rng.random(num_rows) < probability,
        rng.normal(0.0, magnitude, size=num_rows),
        0.0,
    )
    values = np.zeros(num_rows, dtype=np.float64)
    for idx in range(1, num_rows):
        values[idx] = decay * values[idx - 1] + impulses[idx]
    return values


def generate_demo_frame(mode, num_rows=None, seed=20260815, config_path=DEFAULT_CONFIG_PATH):
    """Return one deterministic synthetic navigation or manoeuvring dataset."""
    if mode not in DEFAULT_ROWS:
        raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(DEFAULT_ROWS)}")
    num_rows = DEFAULT_ROWS[mode] if num_rows is None else int(num_rows)
    if num_rows < 2:
        raise ValueError("num_rows must be at least 2")

    rating_kw = _load_rating_kw(config_path)
    rng = np.random.default_rng(seed)
    states, targets = _operating_targets(num_rows, mode, rng)
    response = 0.015 if mode == "navigation" else 0.08
    command_noise = 0.0008 if mode == "navigation" else 0.002
    lever_port = _track_command(targets[:, 0], response, command_noise, rng)
    lever_starboard = _track_command(targets[:, 1], response, command_noise, rng)

    seconds = np.arange(num_rows, dtype=np.float64)
    daily_phase = 2.0 * np.pi * seconds / 86_400.0
    wind_noise = _ar1(num_rows, 0.995, 0.025, rng)
    air_temp = 18.0 + 3.0 * np.sin(daily_phase - 0.8) + _ar1(num_rows, 0.999, 0.015, rng)
    sea_temp = 16.0 + 0.6 * np.sin(daily_phase - 1.5) + _ar1(num_rows, 0.9995, 0.005, rng)
    wind_intensity = np.clip(5.0 + 1.4 * np.sin(daily_phase + 0.4) + wind_noise, 0.0, 18.0)
    sea_state = np.clip(np.rint(0.8 + wind_intensity / 3.5), 0.0, 6.0)

    mean_command = (np.abs(lever_port) + np.abs(lever_starboard)) / 2.0
    imbalance = np.abs(lever_port - lever_starboard)
    command_change = np.r_[0.0, np.abs(np.diff(lever_port)) + np.abs(np.diff(lever_starboard))]
    if mode == "navigation":
        target_power = 0.18 + 0.78 * mean_command + 0.008 * wind_intensity
        power_response = 0.985
        transient = _shot_noise(num_rows, 1.0 / 10_800.0, 0.04, 0.985, rng)
        power_noise = _ar1(num_rows, 0.97, 0.002, rng)
    else:
        target_power = 0.08 + 0.72 * mean_command + 0.20 * imbalance + 1.8 * command_change
        power_response = 0.92
        transient = _shot_noise(num_rows, 1.0 / 600.0, 0.08, 0.90, rng)
        power_noise = _ar1(num_rows, 0.90, 0.004, rng)

    desired_power = target_power + transient + power_noise
    active_power_pu = np.empty(num_rows, dtype=np.float64)
    active_power_pu[0] = desired_power[0]
    for idx in range(1, num_rows):
        active_power_pu[idx] = (
            power_response * active_power_pu[idx - 1]
            + (1.0 - power_response) * desired_power[idx - 1]
        )
    active_power_pu = np.clip(active_power_pu, 0.03, 1.20)

    power_factor = np.clip(
        0.94 - 0.06 * mean_command - 0.025 * imbalance + _ar1(num_rows, 0.995, 0.0005, rng),
        0.78,
        0.98,
    )
    reactive_power_pu = active_power_pu * np.tan(np.arccos(power_factor))
    reactive_power_pu += _ar1(num_rows, 0.95, 0.0015, rng)
    reactive_power_pu = np.clip(reactive_power_pu, 0.0, 0.90)

    if mode == "navigation":
        man_port = np.zeros(num_rows, dtype=np.int8)
        man_starboard = np.zeros(num_rows, dtype=np.int8)
    else:
        man_port = np.isin(states, [1, 2, 3]).astype(np.int8)
        man_starboard = np.isin(states, [1, 2, 4]).astype(np.int8)

    timestamps = pd.date_range("2000-01-01", periods=num_rows, freq="s")
    return pd.DataFrame({
        "date": timestamps.strftime("%Y-%m-%d"),
        "time": timestamps.strftime("%H:%M:%S"),
        "act_pow_kW": (active_power_pu * rating_kw).astype(np.float32),
        "react_pow_kVAR": (reactive_power_pu * rating_kw).astype(np.float32),
        "man_selector_port": man_port,
        "man_selector_starboard": man_starboard,
        "lever_port": lever_port.astype(np.float32),
        "lever_starboard": lever_starboard.astype(np.float32),
        "air_temp": air_temp.astype(np.float32),
        "wind_intensity": wind_intensity.astype(np.float32),
        "sea_temp": sea_temp.astype(np.float32),
        "sea_state": sea_state.astype(np.float32),
    })


def write_demo_dataset(
    mode,
    output_dir,
    num_rows=None,
    seed=20260815,
    force=False,
    config_path=DEFAULT_CONFIG_PATH,
):
    output_path = Path(output_dir) / OUTPUT_NAMES[mode]
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_demo_frame(mode, num_rows=num_rows, seed=seed, config_path=config_path)
    frame.to_parquet(output_path, index=False)
    return output_path, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["navigation", "manoeuvre", "all"], default="all")
    parser.add_argument("--rows", type=int, help="Override the mode-specific default row count.")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="Replace existing demo files.")
    args = parser.parse_args()

    modes = tuple(DEFAULT_ROWS) if args.mode == "all" else (args.mode,)
    for offset, mode in enumerate(modes):
        path, frame = write_demo_dataset(
            mode,
            args.output_dir,
            num_rows=args.rows,
            seed=args.seed + offset,
            force=args.force,
            config_path=args.config,
        )
        print(f"Wrote {len(frame):,} {mode} rows to {path} ({path.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
