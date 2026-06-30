# GEPC

GEPC now uses a lightweight PyTorch + Lightning + Hydra training pipeline.
The repository is organized around two main entrypoints: [`script/simulate.py`](script/simulate.py) for generating simulation data and [`script/train.py`](script/train.py) for inverse-control training.

## Structure

- [`script/`](script): minimal main scripts
- [`notebook/`](notebook): post-training analysis from checkpoints and simulation data
- [`config/pipeline/`](config/pipeline): top-level Hydra pipelines
- [`config/schema/data.yaml`](config/schema/data.yaml): training data loading configuration
- [`config/schema/simulator.yaml`](config/schema/simulator.yaml): simulation data generation configuration
- [`config/schema/model/`](config/schema/model): selectable model groups
- [`config/experiment/simulate/`](config/experiment/simulate): simulation data presets
- [`config/experiment/train/`](config/experiment/train): training experiment presets
- [`src/data.py`](src/data.py): Dataset and LightningDataModule
- [`src/simulator.py`](src/simulator.py): cell simulation and camera model
- [`src/model/`](src/model): LSTM and Transformer inverse-control models
- [`src/objective.py`](src/objective.py): LightningModule objective
- [`src/trainer.py`](src/trainer.py): Lightning trainer wrapper with TensorBoard + checkpoints

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

## Data And Checkpoints

The repository does not track the local [`data/`](data) directory. Precomputed
simulation pools, TensorBoard logs, and trained checkpoints are available on
[OSF](https://osf.io/mnvcz/).

After downloading, place the folder at the repository root so the paths below
exist:

- `data/simulate/`
- `data/train-log/`
- `data/train-ckpt/`

You can download the artifacts from the OSF web page, or use `osfclient`:

```bash
conda install -n gepc -c conda-forge osfclient
osf -p mnvcz clone /tmp/gepc-osf
mkdir -p data
rsync -a /tmp/gepc-osf/osfstorage/data/ data/
```

If the OSF project is private, authenticate with your OSF account:

```bash
export OSF_PASSWORD="your-osf-password"
osf -u your-osf-email -p mnvcz clone /tmp/gepc-osf
unset OSF_PASSWORD
```

## Training

Generate one simulation dataset:

```bash
python -m script.simulate --config-name experiment/simulate/Style4-Train
```

Run the long-range experiment:

```bash
python -m script.train --config-name experiment/train/Style4-Past36-LSTM
```

Run the Transformer counterpart:

```bash
python -m script.train --config-name experiment/train/Style4-Past36-Transformer
```

Generated artifacts are written under [`data/`](data):

- simulation datasets under [`data/simulate/`](data/simulate)
- simulation job logs under [`data/simulate-log/`](data/simulate-log)
- TensorBoard logs under [`data/train-log/`](data/train-log)
- checkpoints under [`data/train-ckpt/`](data/train-ckpt)

## Analysis

Post-training analysis is done in notebooks. The notebooks load trained checkpoints from [`data/train-ckpt/`](data/train-ckpt) and validation data from [`data/simulate/`](data/simulate), then compute predictions and render the final figures.

Current figure notebooks:

- [`notebook/evaluation-Base.ipynb`](notebook/evaluation-Base.ipynb)
- [`notebook/evaluation-ErrorRelateToCosineTargetPeriod.ipynb`](notebook/evaluation-ErrorRelateToCosineTargetPeriod.ipynb)
- [`notebook/ablation-ModelArchitectures.ipynb`](notebook/ablation-ModelArchitectures.ipynb)
- [`notebook/ablation-DataSimulationMethod.ipynb`](notebook/ablation-DataSimulationMethod.ipynb)
- [`notebook/ablation-InputLength.ipynb`](notebook/ablation-InputLength.ipynb)
