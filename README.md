# GEPC

GEPC now uses a lightweight PyTorch + Lightning + Hydra training pipeline.
The repository is organized around one training entrypoint in [`script/pretrain.py`](/Users/tianrui.qi/Documents/GitHub/GEPC/script/pretrain.py), modular source code in [`src/`](/Users/tianrui.qi/Documents/GitHub/GEPC/src), and executed analysis notebooks in [`notebook/`](/Users/tianrui.qi/Documents/GitHub/GEPC/notebook).

## Structure

- [`script/`](/Users/tianrui.qi/Documents/GitHub/GEPC/script): minimal main scripts
- [`config/pipeline/`](/Users/tianrui.qi/Documents/GitHub/GEPC/config/pipeline): top-level Hydra pipelines
- [`config/schema/data.yaml`](/Users/tianrui.qi/Documents/GitHub/GEPC/config/schema/data.yaml): data configuration
- [`config/schema/model/`](/Users/tianrui.qi/Documents/GitHub/GEPC/config/schema/model): selectable model groups
- [`config/experiment/`](/Users/tianrui.qi/Documents/GitHub/GEPC/config/experiment): concrete experiment presets
- [`src/data/`](/Users/tianrui.qi/Documents/GitHub/GEPC/src/data): datasets, simulation backend, LightningDataModule
- [`src/model/`](/Users/tianrui.qi/Documents/GitHub/GEPC/src/model): LSTM and Transformer forecasters
- [`src/objective/`](/Users/tianrui.qi/Documents/GitHub/GEPC/src/objective): LightningModule objectives
- [`src/trainer/`](/Users/tianrui.qi/Documents/GitHub/GEPC/src/trainer): Lightning trainer wrapper with TensorBoard + checkpoints
- [`notebook/`](/Users/tianrui.qi/Documents/GitHub/GEPC/notebook): executed downstream analysis notebooks

## Environment

This project is built around
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)
for training,
[Hydra](https://github.com/facebookresearch/hydra)
for configuration management, and
[Conda](https://docs.conda.io/en/latest/)
for dependency management.
To set up the environment,

```bash
# clone the repository
git clone git@github.com:tianrui-qi/GEPC.git
cd GEPC
# create the conda environment
conda env create -f environment.yaml
conda activate gepc
```

## Training

Run the long-range experiment:

```bash
python -m script.pretrain +experiment=01
```

You can swap Hydra model groups directly from the CLI, and data overrides stay flat:

```bash
python -m script.pretrain +experiment=01 model=transformer data.batch_size=128 trainer.max_epochs=8
```

Artifacts are written by experiment name:

- TensorBoard logs under [`log/`](/Users/tianrui.qi/Documents/GitHub/GEPC/log)
- checkpoints under [`ckpt/`](/Users/tianrui.qi/Documents/GitHub/GEPC/ckpt)

## Analysis

Prediction and experiment analysis live entirely in executed Jupyter notebooks. The repository currently includes one executed analysis notebook:

- [`notebook/analysis01.ipynb`](/Users/tianrui.qi/Documents/GitHub/GEPC/notebook/analysis01.ipynb)
