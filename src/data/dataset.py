from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Sequence

import numpy as np
import torch

from .simulation import SimulationRecord


@dataclass(frozen=True)
class ForecastArrays:
    past: np.ndarray
    future_stim: np.ndarray
    target: np.ndarray
    observed: np.ndarray
    stimulus: np.ndarray
    cell_seed: np.ndarray
    sim_seed: np.ndarray
    measure_seed: np.ndarray
    split: np.ndarray

    @classmethod
    def from_records(
        cls,
        records: Sequence[SimulationRecord],
        *,
        scale: float,
        past_steps: int,
        horizon: int,
        record_indices: np.ndarray,
        window_starts: np.ndarray,
    ) -> "ForecastArrays":
        if not records:
            raise ValueError("records must not be empty")
        if len(record_indices) != len(window_starts):
            raise ValueError("record_indices and window_starts must have the same length")

        samples: list[dict[str, np.ndarray | int | str]] = []
        for record_index, window_start in zip(record_indices, window_starts):
            record = records[int(record_index)]
            sample = build_window_sample(
                record,
                start=int(window_start),
                scale=scale,
                past_steps=past_steps,
                horizon=horizon,
            )
            samples.append(sample)

        return cls(
            past=np.stack([sample["past"] for sample in samples], axis=0),
            future_stim=np.stack([sample["future_stim"] for sample in samples], axis=0),
            target=np.stack([sample["target"] for sample in samples], axis=0),
            observed=np.stack([sample["observed"] for sample in samples], axis=0),
            stimulus=np.stack([sample["stimulus"] for sample in samples], axis=0),
            cell_seed=np.asarray([sample["cell_seed"] for sample in samples], dtype=np.int64),
            sim_seed=np.asarray([sample["sim_seed"] for sample in samples], dtype=np.int64),
            measure_seed=np.asarray([sample["measure_seed"] for sample in samples], dtype=np.int64),
            split=np.asarray([sample["split"] for sample in samples]),
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "past": self.past,
            "future_stim": self.future_stim,
            "target": self.target,
            "observed": self.observed,
            "stimulus": self.stimulus,
            "cell_seed": self.cell_seed,
            "sim_seed": self.sim_seed,
            "measure_seed": self.measure_seed,
            "split": self.split,
        }


def build_window_sample(
    record: SimulationRecord,
    *,
    start: int,
    scale: float,
    past_steps: int,
    horizon: int,
) -> dict[str, np.ndarray | int | str]:
    window_steps = past_steps + horizon
    stop = start + window_steps
    observed = record.observed[start:stop].astype(np.float32)
    stimulus = record.stimulus[start:stop].astype(np.float32)

    return {
        "past": np.column_stack(
            [
                observed[:past_steps] / scale,
                stimulus[:past_steps],
            ]
        ).astype(np.float32),
        "future_stim": stimulus[past_steps:].astype(np.float32),
        "target": (observed[past_steps:] / scale).astype(np.float32),
        "observed": observed,
        "stimulus": stimulus,
        "cell_seed": record.cell_seed,
        "sim_seed": record.sim_seed,
        "measure_seed": record.measure_seed,
        "split": record.split,
    }


class ForecastDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: Sequence[SimulationRecord],
        *,
        scale: float,
        past_steps: int,
        horizon: int,
        samples_per_cell: int,
        seed: int,
        random_windows: bool,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")

        self.records = tuple(records)
        self.scale = scale
        self.past_steps = past_steps
        self.horizon = horizon
        self.samples_per_cell = samples_per_cell
        self.random_windows = random_windows
        self.window_steps = past_steps + horizon

        self.record_indices = np.repeat(
            np.arange(len(self.records), dtype=np.int64),
            max(samples_per_cell, 1),
        )
        self.window_starts = self._build_window_starts(seed)
        self.forecast_arrays = ForecastArrays.from_records(
            self.records,
            scale=scale,
            past_steps=past_steps,
            horizon=horizon,
            record_indices=self.record_indices,
            window_starts=self.window_starts,
        )
        self.arrays = self.forecast_arrays.as_dict()

    def _max_window_start(self, record_index: int) -> int:
        total_steps = int(self.records[record_index].observed.shape[0])
        return max(total_steps - self.window_steps, 0)

    def _build_window_starts(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        window_starts = np.zeros(len(self.record_indices), dtype=np.int64)
        for index, record_index in enumerate(self.record_indices):
            max_start = self._max_window_start(int(record_index))
            if max_start > 0:
                window_starts[index] = int(rng.integers(0, max_start + 1))
        return window_starts

    def _resolve_window_start(self, index: int, record_index: int) -> int:
        if not self.random_windows:
            return int(self.window_starts[index])

        max_start = self._max_window_start(record_index)
        if max_start == 0:
            return 0
        return int(torch.randint(0, max_start + 1, (1,)).item())

    def __len__(self) -> int:
        return int(self.record_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index = int(self.record_indices[index])
        start = self._resolve_window_start(index, record_index)
        sample = build_window_sample(
            self.records[record_index],
            start=start,
            scale=self.scale,
            past_steps=self.past_steps,
            horizon=self.horizon,
        )
        return {
            "past": torch.as_tensor(sample["past"], dtype=torch.float32),
            "future_stim": torch.as_tensor(sample["future_stim"], dtype=torch.float32),
            "target": torch.as_tensor(sample["target"], dtype=torch.float32),
            "index": index,
        }
