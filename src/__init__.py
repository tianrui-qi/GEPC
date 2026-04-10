import os
import platform

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .data import DataModule
from .model import Model
from .objective import ObjectivePretrain
from .trainer import Trainer
from .utils import (
    REPO_ROOT,
    compose_experiment_config,
    load_model_from_checkpoint,
    resolve_repo_path,
    resolve_runtime_paths,
    save_config,
)

__all__ = [
    "DataModule",
    "Model",
    "ObjectivePretrain",
    "Trainer",
    "REPO_ROOT",
    "compose_experiment_config",
    "load_model_from_checkpoint",
    "resolve_repo_path",
    "resolve_runtime_paths",
    "save_config",
]
