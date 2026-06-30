from pathlib import Path
import re
import sys

import hydra
import hydra.utils
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import torch

from src.data import Dataset
from src.objective import DifferentiableODE


CAMERA_MAX = 4095.0
EVAL_STEPS = 288
FREQ_SWEEP_EVAL_STEPS = 480
FREQ_PERIODS_STEPS = [48, 72, 96, 144, 192]
CANONICAL_PERIOD = 96
CANONICAL_H = 8.0
COSINE_MID_AU = 1500.0
COSINE_AMP_AU = 750.0
DEMO_HOURS = 24.0
SMOOTH_WINDOW = 6
STEP_MIN = 5


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "config"
ASSET_DIR = REPO / "asset"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.family": "sans-serif",
})
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


_CONFIG_CACHE = {}
_MODEL_CACHE = {}
_POOL_CACHE = {}


def _experiment_config_name(experiment_name: str) -> str:
    if experiment_name.startswith("experiment/"):
        return experiment_name
    if experiment_name.startswith("train/") or experiment_name.startswith("simulate/"):
        return f"experiment/{experiment_name}"
    return f"experiment/train/{experiment_name}"


def compose_experiment_config(experiment_name: str):
    if experiment_name not in _CONFIG_CACHE:
        with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            _CONFIG_CACHE[experiment_name] = hydra.compose(
                config_name=_experiment_config_name(experiment_name)
            )
    return _CONFIG_CACHE[experiment_name]


def load_model_weights(model: torch.nn.Module, ckpt_path: str | Path) -> torch.nn.Module:
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.replace("model.", "", 1).replace("network.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state, strict=False)
    return model


def load_model(
    experiment_name: str,
    checkpoint: str = "best",
    device: str = "cpu",
) -> torch.nn.Module:
    cache_key = (experiment_name, checkpoint, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    cfg = compose_experiment_config(experiment_name)
    ckpt_path = Path(cfg.trainer.ckpt_save_fold) / f"{checkpoint}.ckpt"
    if not ckpt_path.is_absolute():
        ckpt_path = REPO / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = hydra.utils.instantiate(cfg.model)
    load_model_weights(model, ckpt_path)
    model.to(torch.device(device))
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def load_ode(experiment_name: str) -> DifferentiableODE:
    return DifferentiableODE(**compose_experiment_config(experiment_name).simulator)


def load_eval_pool(style: int = 4, n: int | None = None):
    cache_key = (style, n)
    if cache_key in _POOL_CACHE:
        return _POOL_CACHE[cache_key]

    path = REPO / "data" / "simulate" / f"Style{style}-Valid.pkl"
    records = Dataset.load_records(str(path))
    if n is not None:
        records = records[:n]
    _POOL_CACHE[cache_key] = records
    return records


def experiment_runtime(experiment_name: str):
    cfg = compose_experiment_config(experiment_name)
    model = load_model(experiment_name)
    ode = load_ode(experiment_name)
    horizon = int(cfg.data.horizon)
    past_steps = int(cfg.data.past_steps)
    camera_max = float(cfg.simulator.camera_max)
    return cfg, model, ode, horizon, past_steps, camera_max


def make_cosine_target(
    period_steps: int,
    n_steps: int,
    camera_max: float = CAMERA_MAX,
) -> np.ndarray:
    time_index = np.arange(n_steps, dtype=np.float32)
    wave = COSINE_MID_AU + COSINE_AMP_AU * np.cos(
        2 * np.pi * time_index / period_steps
    )
    return np.clip(wave, 0.0, camera_max).astype(np.float32)


def rising_mask(target: np.ndarray) -> np.ndarray:
    diff = np.diff(target, prepend=target[0])
    return diff > 0


@torch.no_grad()
def predict_stimulus(
    model: torch.nn.Module,
    past_obs_scaled: np.ndarray,
    past_stimulus: np.ndarray,
    target_scaled: np.ndarray,
) -> np.ndarray:
    past = torch.as_tensor(
        np.column_stack([past_obs_scaled, past_stimulus])[None].astype(np.float32)
    )
    target = torch.as_tensor(target_scaled[None].astype(np.float32))
    return torch.sigmoid(model(past, target)).squeeze(0).numpy()


def run_ode_mpc(
    model: torch.nn.Module,
    warmup_obs: np.ndarray,
    warmup_stimulus: np.ndarray,
    desired_au: np.ndarray,
    cell_e: float,
    horizon: int,
    ode: DifferentiableODE,
    camera_max: float = CAMERA_MAX,
    context_steps: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    cell_e_tensor = torch.tensor([cell_e], dtype=torch.float32)
    past_stimulus = torch.as_tensor(warmup_stimulus[None], dtype=torch.float32)
    last_obs = torch.tensor([warmup_obs[-1] / camera_max], dtype=torch.float32)
    hidden_h, _ = ode.simulate_past(past_stimulus, cell_e_tensor)
    hidden_f = ode.obs_to_molecules(last_obs)

    history_obs = list(warmup_obs.astype(np.float32))
    history_stimulus = list(warmup_stimulus.astype(np.float32))
    recent_stimulus = list(warmup_stimulus[-ode.tau_steps:].astype(np.float32))

    ode_obs = []
    applied_stimulus = []
    for block_start in range(0, len(desired_au), horizon):
        block_end = min(block_start + horizon, len(desired_au))
        block_len = block_end - block_start
        past_obs = np.array(history_obs[-context_steps:], dtype=np.float32)
        past_stimulus = np.array(history_stimulus[-context_steps:], dtype=np.float32)
        target_au = np.pad(
            desired_au[block_start:block_end],
            (0, horizon - block_len),
            mode="edge",
        )
        stim_hard = (
            predict_stimulus(
                model,
                past_obs / camera_max,
                past_stimulus,
                target_au / camera_max,
            )
            > 0.5
        ).astype(np.float32)

        for local_index in range(block_len):
            light_input = torch.tensor([recent_stimulus[0]], dtype=torch.float32)
            hidden_h, hidden_f = ode.step(hidden_h, hidden_f, light_input, cell_e_tensor)
            observed_au = float(ode.fluorescence_to_au(hidden_f).item())
            ode_obs.append(observed_au)
            applied_stimulus.append(stim_hard[local_index])
            recent_stimulus.pop(0)
            recent_stimulus.append(float(stim_hard[local_index]))
            history_obs.append(observed_au)
            history_stimulus.append(float(stim_hard[local_index]))

    return np.array(ode_obs, dtype=np.float32), np.array(applied_stimulus, dtype=np.float32)


def _figure_path(title: str, save_name: str | None = None) -> Path:
    stem = save_name or re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
    return ASSET_DIR / f"{stem}.png"


def show_figure(title: str, save_name: str | None = None) -> Path:
    fig = plt.gcf()
    save_path = _figure_path(title, save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
    relative_path = save_path.relative_to(REPO).as_posix()
    markdown = f"saved figure: [{relative_path}](../{relative_path})"
    try:
        from IPython.display import Markdown, display

        display(Markdown(markdown))
    except ImportError:
        print(markdown)
    plt.close(fig)
    return save_path


def smooth(values, window: int = SMOOTH_WINDOW) -> np.ndarray:
    return uniform_filter1d(np.array(values, dtype=float), size=window, mode="nearest")


def shade_falling_phases(
    ax,
    times,
    period_h: float = CANONICAL_H,
) -> None:
    t_max = float(times[-1])
    for index in range(int(t_max / period_h) + 2):
        start = index * period_h
        end = start + period_h / 2
        start_clipped = max(start, 0)
        end_clipped = min(end, t_max)
        if start_clipped < end_clipped:
            ax.axvspan(
                start_clipped,
                end_clipped,
                alpha=0.06,
                color="tomato",
                linewidth=0,
            )


def draw_period_ticks(
    ax,
    period_h: float = CANONICAL_H,
    t_max: float = 24.0,
) -> None:
    tick = period_h
    while tick <= t_max:
        ax.axvline(tick, color="gray", lw=0.6, ls=":", alpha=0.5)
        tick += period_h


def compute_rollout_vs_time(
    experiment_name: str,
    eval_style: int = 4,
    period_steps: int = CANONICAL_PERIOD,
    eval_steps: int = EVAL_STEPS,
) -> dict:
    cfg, model, ode, horizon, past_steps, camera_max = experiment_runtime(experiment_name)
    desired = make_cosine_target(period_steps, eval_steps, camera_max)
    all_errors = []
    for record in load_eval_pool(style=eval_style):
        ode_response, _ = run_ode_mpc(
            model,
            record.observed[:past_steps],
            record.stimulus[:past_steps],
            desired,
            record.cell_E,
            horizon,
            ode,
            camera_max,
            context_steps=36,
        )
        all_errors.append(np.abs(ode_response - desired) / camera_max)

    all_errors = np.array(all_errors)
    return {
        "times": np.array([round(step * STEP_MIN / 60, 3) for step in range(eval_steps)]),
        "mean": np.mean(all_errors, axis=0),
        "std": np.std(all_errors, axis=0),
    }


def plot_error_time(
    configs,
    title: str,
    save_name: str | None = None,
) -> None:
    loaded = [
        (label, color, compute_rollout_vs_time(experiment))
        for label, color, experiment in configs
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    shade_falling_phases(ax, loaded[0][2]["times"])
    draw_period_ticks(ax, t_max=float(loaded[0][2]["times"][-1]))

    handles = []
    for label, color, rollout in loaded:
        times = rollout["times"]
        mean = smooth(rollout["mean"])
        std = smooth(rollout["std"])
        ax.fill_between(times, np.maximum(mean - std, 0), mean + std, alpha=0.12, color=color)
        line, = ax.plot(
            times,
            mean,
            color=color,
            lw=2,
            label=f'{label} (avg={np.mean(rollout["mean"]):.4f})',
        )
        handles.append(line)

    handles.append(mpatches.Patch(color="tomato", alpha=0.25, label="Falling phase"))
    ax.set_xlabel("Time into rollout (hr)")
    ax.set_ylabel("nMAE")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    plt.tight_layout()
    show_figure(title, save_name)


def trajectory_to_dict(
    warmup_obs: np.ndarray,
    warmup_stimulus: np.ndarray,
    ode_response: np.ndarray,
    applied_stimulus: np.ndarray,
    target: np.ndarray,
    past_steps: int,
) -> dict:
    return {
        "times_warmup_hr": [round(i * STEP_MIN / 60, 4) for i in range(-len(warmup_obs), 0)],
        "times_eval_hr": [round(i * STEP_MIN / 60, 4) for i in range(len(ode_response))],
        "warmup_obs_au": warmup_obs.tolist(),
        "warmup_stim": warmup_stimulus.tolist(),
        "ode_resp_au": ode_response.tolist(),
        "applied_stim": applied_stimulus.tolist(),
        "target_au": target.tolist(),
        "past_steps": past_steps,
    }


def compute_trajectory(
    experiment_name: str,
    eval_style: int = 4,
    demo_cells: int = 3,
    period_steps: int = CANONICAL_PERIOD,
    eval_steps: int = EVAL_STEPS,
) -> dict:
    cfg, model, ode, horizon, past_steps, camera_max = experiment_runtime(experiment_name)
    target = make_cosine_target(period_steps, eval_steps, camera_max)
    cells = []
    for record in load_eval_pool(style=eval_style, n=demo_cells):
        warmup_obs = record.observed[:past_steps]
        warmup_stimulus = record.stimulus[:past_steps]
        ode_response, applied_stimulus = run_ode_mpc(
            model,
            warmup_obs,
            warmup_stimulus,
            target,
            record.cell_E,
            horizon,
            ode,
            camera_max,
            context_steps=past_steps,
        )
        cells.append(
            trajectory_to_dict(
                warmup_obs,
                warmup_stimulus,
                ode_response,
                applied_stimulus,
                target,
                past_steps,
            )
        )
    return {"cells": cells, "experiment_name": experiment_name}


def load_trajectory(
    experiment_name: str,
    cell_index: int = 0,
) -> dict:
    return compute_trajectory(experiment_name)["cells"][cell_index]


def plot_trajectory_group(
    configs,
    title: str,
    save_name: str | None = None,
) -> None:
    loaded = [
        (label, color, load_trajectory(experiment))
        for label, color, experiment in configs
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    reference = loaded[0][2]
    warmup_times = np.array(reference["times_warmup_hr"])
    eval_times = np.array(reference["times_eval_hr"])
    target = np.array(reference["target_au"])
    start_time = float(warmup_times[0]) if len(warmup_times) else 0.0
    end_time = float(eval_times[-1])

    shade_falling_phases(ax, eval_times)
    draw_period_ticks(ax, t_max=end_time)
    ax.axvspan(start_time, 0, alpha=0.08, color="steelblue", linewidth=0)
    ax.axvline(0, color="steelblue", lw=1.2, ls="--", alpha=0.6)
    ax.plot(eval_times, target, color="black", lw=2, ls="--", label="Target", zorder=5)

    handles = []
    for label, color, trajectory in loaded:
        ax.plot(
            trajectory["times_warmup_hr"],
            trajectory["warmup_obs_au"],
            color=color,
            lw=1.2,
            alpha=0.35,
        )
        line, = ax.plot(
            trajectory["times_eval_hr"],
            trajectory["ode_resp_au"],
            color=color,
            lw=2,
            label=label,
        )
        handles.append(line)

    legend_handles = [
        matplotlib.lines.Line2D([0], [0], color="black", lw=2, ls="--", label="Target"),
        *handles,
        mpatches.Patch(color="tomato", alpha=0.25, label="Falling phase"),
        mpatches.Patch(color="steelblue", alpha=0.2, label="Warmup"),
    ]
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Fluorescence (AU)")
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(bottom=0)
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")
    plt.tight_layout()
    show_figure(title, save_name)


def plot_single_rollout_trajectory(
    experiment_name: str,
    title: str,
    save_name: str | None = None,
) -> None:
    trajectory = load_trajectory(experiment_name)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    times = np.array(trajectory["times_eval_hr"])
    mask = times <= DEMO_HOURS

    ax.plot(
        trajectory["times_warmup_hr"],
        trajectory["warmup_obs_au"],
        color="gray",
        lw=1,
        alpha=0.5,
        label="Warmup",
    )
    ax.axvline(0, color="gray", lw=1, ls=":", alpha=0.7)
    ax.plot(
        times[mask],
        np.array(trajectory["target_au"])[mask],
        "k--",
        lw=1.5,
        label="Target",
    )
    ax.plot(
        times[mask],
        np.array(trajectory["ode_resp_au"])[mask],
        color=COLORS[0],
        lw=1.8,
        label="LSTM MPC",
    )

    applied = np.array(trajectory["applied_stim"])[mask]
    plotted_times = times[mask]
    for index, stimulus in enumerate(applied):
        if stimulus > 0.5:
            ax.axvspan(
                plotted_times[index],
                plotted_times[index] + STEP_MIN / 60,
                alpha=0.12,
                color="green",
                linewidth=0,
            )

    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("GFP fluorescence (AU)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, CAMERA_MAX * 0.7)
    ax.legend(fontsize=9)
    plt.tight_layout()
    show_figure(title, save_name)
