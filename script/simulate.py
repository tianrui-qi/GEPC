"""
Generate one simulation dataset.

Examples:
    python -m script.simulate --config-name experiment/simulate/Style4-Train
    python -m script.simulate simulator.seed=2000 simulator.num_cells=10
"""
import pickle
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.simulator import simulate


def _bounded_cumsum_stims(
    total_chambers: int,
    timepoints: int,
    *,
    upper_limit: int,
    lower_limit: int,
) -> np.ndarray:
    flips = np.round(np.random.rand(total_chambers, timepoints))
    flips[flips == 0] = -1

    sums = np.zeros((total_chambers, timepoints), dtype=np.float32)
    sums[:, 0] = np.floor((upper_limit + lower_limit) / 2)
    for time_index in range(timepoints - 1):
        sums[:, time_index + 1] = sums[:, time_index] + flips[:, time_index]
        sums = sums.clip(lower_limit, upper_limit)
    return sums >= 0


def random_stimulus(
    *,
    total_steps: int,
    nostim_steps: int,
    seed: int,
    stim_config_index: int,
    stim_configs: list,
) -> np.ndarray:
    upper_limit, lower_limit = stim_configs[stim_config_index % len(stim_configs)]
    state = np.random.get_state()
    np.random.seed(seed)
    stimulus = _bounded_cumsum_stims(
        1,
        total_steps,
        upper_limit=upper_limit,
        lower_limit=lower_limit,
    )[0]
    stimulus[:nostim_steps] = 0
    np.random.set_state(state)
    return stimulus.astype(np.float32)


def generate_records(cfg: DictConfig):
    simulator = cfg.simulator
    rng = np.random.default_rng(int(simulator.seed))
    num_cells = int(simulator.num_cells)

    cell_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(num_cells)]
    stim_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(num_cells)]
    sim_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(num_cells)]
    measure_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(num_cells)]

    records = []
    for index in tqdm(range(num_cells), desc="simulate", unit="cell"):
        stimulus = random_stimulus(
            total_steps=int(simulator.steps),
            nostim_steps=int(simulator.nostim_steps),
            seed=stim_seeds[index],
            stim_config_index=int(simulator.stim_config_index),
            stim_configs=list(simulator.stim_configs),
        )
        records.append(
            simulate(
                cell_class=str(simulator.cell_class),
                cell_seed=cell_seeds[index],
                stimulus=stimulus,
                solver=str(simulator.solver),
                camera_noise_perc=float(simulator.camera_noise_perc),
                sim_seed=sim_seeds[index],
                measure_seed=measure_seeds[index],
                sampling=int(simulator.sampling),
                camera_mult=float(simulator.camera_mult),
                camera_offset=float(simulator.camera_offset),
                camera_max=float(simulator.camera_max),
                eta=float(simulator.eta),
                nu=float(simulator.nu),
                h1=float(simulator.h1),
                h2=float(simulator.h2),
                a=float(simulator.a),
                k_h=float(simulator.k_h),
                n_h=float(simulator.n_h),
                tau=float(simulator.tau),
                mu=float(simulator.mu),
                sigma=float(simulator.sigma),
            )
        )
    return records


@hydra.main(version_base=None, config_path="../config", config_name="pipeline/simulate")
def main(cfg: DictConfig) -> None:
    data_save_path = Path(cfg.simulator.data_save_path)
    if data_save_path.exists() and not bool(cfg.simulator.overwrite):
        print(f"[skip] {data_save_path} already exists; set simulator.overwrite=true to replace it")
        return

    data_save_path.parent.mkdir(parents=True, exist_ok=True)
    records = generate_records(cfg)
    with open(data_save_path, "wb") as file:
        pickle.dump(records, file)
    print(f"saved {len(records)} records -> {data_save_path}")


if __name__ == "__main__":
    main()
