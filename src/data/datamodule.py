from __future__ import annotations

from dataclasses import dataclass

import lightning
import torch

from .dataset import ForecastDataset
from .simulation import (
    SimulationConfig,
    SimulationSplitConfig,
    generate_simulation_records,
)


@dataclass(frozen=True)
class DataSplitSpec:
    name: str
    num_cells: int
    samples_per_cell: int
    seed_offset: int


class DataModule(lightning.LightningDataModule):
    def __init__(
        self,
        # simulator
        cell_class: str,
        solver: str,
        camera_noise_perc: float,
        scale: float,
        # window
        past_steps: int,
        horizon: int,
        sim_steps: int,
        nostim_steps: int,
        # loader
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        # split
        train_cells: int,
        val_cells: int,
        samples_per_cell: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.past_steps = past_steps
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed

        self.simulation = SimulationConfig(
            cell_class=cell_class,
            solver=solver,
            camera_noise_perc=camera_noise_perc,
            past_steps=past_steps,
            horizon=horizon,
            sim_steps=sim_steps,
            nostim_steps=nostim_steps,
        )
        self.train_split = DataSplitSpec(
            name="train",
            num_cells=train_cells,
            samples_per_cell=samples_per_cell,
            seed_offset=11,
        )
        self.val_split = DataSplitSpec(
            name="valid",
            num_cells=val_cells,
            samples_per_cell=samples_per_cell,
            seed_offset=29,
        )

        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0

        self.train_dataset: torch.utils.data.Dataset | None = None
        self.val_dataset: torch.utils.data.Dataset | None = None

    def _build_dataset(self, split: DataSplitSpec) -> ForecastDataset:
        records = generate_simulation_records(
            simulation=self.simulation,
            split=SimulationSplitConfig(
                name=split.name,
                num_cells=split.num_cells,
                seed=self.seed + split.seed_offset,
            ),
        )
        return ForecastDataset(
            records,
            scale=self.scale,
            past_steps=self.past_steps,
            horizon=self.horizon,
            samples_per_cell=split.samples_per_cell,
            seed=self.seed + split.seed_offset,
            random_windows=split.name == "train",
        )

    def _make_dataloader(
        self,
        dataset: ForecastDataset,
        *,
        shuffle: bool,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None and stage in (None, "fit", "validate"):
            self.train_dataset = self._build_dataset(self.train_split)
            self.val_dataset = self._build_dataset(self.val_split)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset has not been initialized")
        return self._make_dataloader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset has not been initialized")
        return self._make_dataloader(self.val_dataset, shuffle=False)
