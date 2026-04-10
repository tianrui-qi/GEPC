from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from . import simulations as cell_simulations


SAMPLING_MINUTES = int(getattr(cell_simulations, "SAMPLING", 5))


@dataclass(frozen=True)
class SimulationConfig:
    cell_class: str
    solver: str
    camera_noise_perc: float
    past_steps: int
    horizon: int
    sim_steps: int
    nostim_steps: int

    @property
    def total_steps(self) -> int:
        return self.past_steps + self.horizon


@dataclass(frozen=True)
class SimulationSplitConfig:
    name: str
    num_cells: int
    seed: int


@dataclass(frozen=True)
class SimulationRecord:
    observed: np.ndarray
    stimulus: np.ndarray
    cell_seed: int
    sim_seed: int
    measure_seed: int
    split: str


@contextmanager
def numpy_seed(seed: int) -> None:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def new_cell_serial(cell_class: str, seed: int) -> dict:
    with numpy_seed(seed):
        cell = getattr(cell_simulations, cell_class)()
    return cell.serialize()


def clone_cell(serial: dict, cell_class: str):
    cell = getattr(cell_simulations, cell_class)()
    cell.load(copy.deepcopy(serial))
    return cell


def simulate_observed_trajectory(
    serial: dict,
    stims: np.ndarray,
    *,
    cell_class: str,
    solver: str,
    camera_noise_perc: float,
    sim_seed: int,
    measure_seed: int,
) -> np.ndarray:
    stims = stims.astype(np.float32)
    with numpy_seed(sim_seed):
        cell = clone_cell(serial, cell_class)
        cell.set_light_events(stims.astype(float))
        timeseries = cell.run(
            cell.time + len(stims) * SAMPLING_MINUTES,
            solver=solver,
        )
        fluo = np.array([point["F"] for point in timeseries[1:]], dtype=np.float32)[-len(stims) :]
    with numpy_seed(measure_seed):
        observed = cell_simulations.camera_sim(fluo, noise_perc=camera_noise_perc)
    return observed.astype(np.float32)


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


def random_stim_sequences(
    *,
    samples: int,
    total_steps: int,
    nostim_steps: int,
    seed: int,
) -> np.ndarray:
    with numpy_seed(seed):
        stim_configs = [
            (3, -2),
            (1, -2),
            (0, -2),
            (3, -3),
            (2, -3),
            (4, -3),
            (1, -3),
        ]
        chunk_size = max(samples // len(stim_configs), 1)
        chunks: list[np.ndarray] = []
        remaining = samples
        config_index = 0

        while remaining > 0:
            upper_limit, lower_limit = stim_configs[config_index % len(stim_configs)]
            count = min(remaining, chunk_size)
            chunks.append(
                _bounded_cumsum_stims(
                    count,
                    total_steps,
                    upper_limit=upper_limit,
                    lower_limit=lower_limit,
                )
            )
            remaining -= count
            config_index += 1

        stims = np.concatenate(chunks, axis=0) if chunks else np.empty((0, total_steps), dtype=bool)
        np.random.shuffle(stims)
        stims[:, :nostim_steps] = 0
    return stims.astype(np.float32)


def sample_cell_seeds(
    *,
    num_cells: int,
    rng: np.random.Generator,
) -> list[int]:
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(max(num_cells, 1))]


def generate_simulation_records(
    *,
    simulation: SimulationConfig,
    split: SimulationSplitConfig,
) -> list[SimulationRecord]:
    rng = np.random.default_rng(split.seed)
    cell_seeds = sample_cell_seeds(
        num_cells=split.num_cells,
        rng=rng,
    )
    cell_serials = [
        new_cell_serial(simulation.cell_class, cell_seed)
        for cell_seed in cell_seeds
    ]
    return _generate_long_trajectory_records(
        simulation=simulation,
        rng=rng,
        cell_seeds=cell_seeds,
        cell_serials=cell_serials,
        split=split,
    )


def _generate_long_trajectory_records(
    *,
    simulation: SimulationConfig,
    split: SimulationSplitConfig,
    rng: np.random.Generator,
    cell_seeds: list[int],
    cell_serials: list[dict],
) -> list[SimulationRecord]:
    """Simulate one long trajectory per cell and store it in full.

    Window cropping is deferred to the dataset so that every __getitem__ call
    samples a fresh random window (on-the-fly augmentation across epochs).
    Returns one SimulationRecord per cell with observed/stimulus of length sim_steps.
    """
    if simulation.sim_steps < simulation.total_steps:
        raise ValueError(
            "sim_steps must be at least past_steps + horizon for long-trajectory sampling"
        )

    records: list[SimulationRecord] = []

    for cell_seed, serial in tqdm(
        zip(cell_seeds, cell_serials),
        total=len(cell_seeds),
        desc=f"simulation({split.name})",
        unit="cell",
    ):
        stim_seed = int(rng.integers(0, 2**31 - 1))
        long_stims = random_stim_sequences(
            samples=1,
            total_steps=simulation.sim_steps,
            nostim_steps=simulation.nostim_steps,
            seed=stim_seed,
        )[0]

        sim_seed = int(rng.integers(0, 2**31 - 1))
        measure_seed = int(rng.integers(0, 2**31 - 1))
        long_observed = simulate_observed_trajectory(
            serial,
            long_stims,
            cell_class=simulation.cell_class,
            solver=simulation.solver,
            camera_noise_perc=simulation.camera_noise_perc,
            sim_seed=sim_seed,
            measure_seed=measure_seed,
        )

        records.append(
            SimulationRecord(
                observed=long_observed.astype(np.float32),
                stimulus=long_stims.astype(np.float32),
                cell_seed=cell_seed,
                sim_seed=sim_seed,
                measure_seed=measure_seed,
                split=split.name,
            )
        )

    return records
