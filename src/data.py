import pickle
from pathlib import Path
from types import SimpleNamespace

import lightning
import numpy as np
import torch


__all__ = ["DataModule", "Dataset"]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        camera_max: float,
        past_steps: int,
        horizon: int,
        samples_per_cell: int,
        seed: int,
        random_windows: bool,
    ) -> None:
        self.records = tuple(self.load_records(data_path))
        self.camera_max = camera_max
        self.past_steps = past_steps
        self.horizon = horizon
        self.samples_per_cell = samples_per_cell
        self.random_windows = random_windows
        self.window_steps = past_steps + horizon
        self.record_indices = np.repeat(
            np.arange(len(self.records), dtype=np.int64),
            max(samples_per_cell, 1),
        )
        rng = np.random.default_rng(seed)
        self.window_starts = np.array(
            [
                rng.integers(0, self._max_start(int(record_index)) + 1)
                if self._max_start(int(record_index)) > 0 else 0
                for record_index in self.record_indices
            ],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(self.record_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        record_index = int(self.record_indices[index])
        record = self.records[record_index]
        start = self._window_start(index, record_index)
        stop = start + self.window_steps
        observed = record.observed[start:stop].astype(np.float32)
        stimulus = record.stimulus[start:stop].astype(np.float32)

        past = np.column_stack(
            [
                observed[: self.past_steps] / self.camera_max,
                stimulus[: self.past_steps],
            ]
        )
        target = observed[self.past_steps:] / self.camera_max

        return {
            "past": torch.as_tensor(past, dtype=torch.float32),
            "target": torch.as_tensor(target, dtype=torch.float32),
            "cell_E": torch.as_tensor(getattr(record, "cell_E", 40.0), dtype=torch.float32),
            "index": index,
        }

    def _max_start(self, record_index: int) -> int:
        return max(int(self.records[record_index].observed.shape[0]) - self.window_steps, 0)

    def _window_start(self, index: int, record_index: int) -> int:
        if not self.random_windows:
            return int(self.window_starts[index])
        max_start = self._max_start(record_index)
        if max_start == 0:
            return 0
        return int(torch.randint(0, max_start + 1, (1,)).item())

    @staticmethod
    def load_records(data_path: str):
        path = Path(data_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        with open(path, "rb") as file:
            return Dataset._Unpickler(file).load()

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            old_modules = {
                "src.data.simulation",
                "src.data.simulate",
                "src.data.simulations",
                "src.simulations",
                "src.simulator",
            }
            if module in old_modules and name == "SimulationRecord":
                return SimpleNamespace
            return super().find_class(module, name)


class DataModule(lightning.LightningDataModule):
    def __init__(
        self,
        train_load_path: str,
        valid_load_path: str,
        camera_max: float,
        past_steps: int,
        horizon: int,
        samples_per_cell: int,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.train_load_path = train_load_path
        self.valid_load_path = valid_load_path
        self.camera_max = camera_max
        self.past_steps = past_steps
        self.horizon = horizon
        self.samples_per_cell = samples_per_cell
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = num_workers > 0
        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        self.train_dataset = Dataset(
            self.train_load_path,
            self.camera_max,
            self.past_steps,
            self.horizon,
            self.samples_per_cell,
            self.seed + 11,
            True,
        )
        self.valid_dataset = Dataset(
            self.valid_load_path,
            self.camera_max,
            self.past_steps,
            self.horizon,
            self.samples_per_cell,
            self.seed + 29,
            False,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
