# gridutils

Helpers for setting up the environment and submitting DINO training jobs to the SDCC HTCondor GPU pool.

## Directory structure (SDCC)

Both your `$HOME` and `/gpfs01/lbne/users/fm` (shared group area) are visible from the work nodes. However, quotas are very different. The suggested directory layout is:

- `$HOME/ml-dune-model`: clone the repo here. Keep all code in `$HOME`.
- `/gpfs01/lbne/users/fm/${USER}/`: your personal area on the group GPFS volume. Create it once. Inside it:
  - `/gpfs01/lbne/users/fm/${USER}/uvenv/`: python virtual environment. Lives here because it can get large (~10GB).
  - `/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT/`: training run outputs (checkpoints, debug, condor logs). Subdirectories are created automatically by `submit.sh` for each run.
  - `/gpfs01/lbne/users/fm/${USER}/cache/`: will store `warpconvnet/` and `data/` caches.
- `/gpfs01/lbne/users/fm/cffm-data/`: **shared** dataset area, available to anyone. Point `datadir` in your config here.

## Setting up the environment

One-time setup is handled by [build_env.sh](build_env.sh) and [build.sub](build.sub). 
Some packages can be installed directly from the interactive node (thanks to pre-built wheels), but GPU availability is needed when building from source and so the script must be run as a job (via `build.sub`).
See the comments in the script, and the instructions below:

```bash
# 0. get uv package
pip install uv

# 1. create the venv (only needed once)
uv venv /gpfs01/lbne/users/fm/${USER}/uvenv --python 3.11

# 2. CPU-only installs via script
./build_env.sh

# 2b. GPU installs (needed for flash-attn / local warpconvnet builds)
# edit build_env.sh to uncomment the relevant blocks, then:
condor_submit build.sub
```

Edit the env vars at the top of `build_env.sh` (`CUDA`, `TORCH_REL`, `WARPCONV_REL`, `USER`) to match your target stack before running.

## Submitting a training job

There are two relevant files: 
- [trainjob.sh](trainjob.sh): training script that runs on the worker node
- [submit.sh](submit.sh): submission script that prepares the `.sub` file and runs `condor_submit`.

Training jobs can be submitted by:

```bash
./submit.sh path/to/run_config.json
```

What it does:

1. Reads `run_name` from the JSON configuration; this is the campaign name and **must be unique**.
2. Creates `${CONDOR_OUT}/${run_name}/` (errors out if it already exists, always pick a fresh `run_name`).
3. Writes `${run_name}.sub` and submits it.
4. On the worker, [trainjob.sh](trainjob.sh) writes checkpoints/debug to `$_CONDOR_SCRATCH_DIR` and `rsync`s them back to `${CONDOR_OUT}/${run_name}/{checkpoints,debug}/` on exit (including SIGTERM).

### Job submission parameters

At the top of `submit.sh`, you can customize the directory locations as well as the job requirements for your case. Note that the dataset directory is specified directly in the `config.json` file (see below).

```bash
# output base directory on GPFS
CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"

# code directory
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"

# python virtual environment
PYENV="${PYENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"

# cache directory for warpconvnet and data index
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"

# JOB REQUIREMENTS: memory, GPU type, etc.
REQUEST_MEMORY="${REQUEST_MEMORY:-32000}"
REQUEST_GPUS="${REQUEST_GPUS:-1}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"
```

### Training configuration

An example training configuration is provided in [config.json](config.json).
The full list of accepted keys is whatever `dino.train_dino.from_config` consumes, see [dino/train_dino.py](../dino/train_dino.py).

Copy [config.json](config.json) and edit. Key fields:

- `run_name` — **must be unique**; defines the output directory under `CONDOR_OUT`.
- `datadir`, `apa`, `view`, `image_h`, `image_w`, `n_subset` — dataset selection.
- `batch_size`, `num_workers` — dataloader.
- `backbone_name`, `feature_dim`, `proj_head_*`, `encoding_range` — model.
- `augmentation_mode`, `crop_*`, `mask_ratio` — augmentation pipeline.
- `epochs`, `lr`, `min_lr`, `weight_decay*`, `warmup_epochs`, `momentum_*` — schedule.
- `loss_type`, `teacher_temp`, `student_temp`, `use_centering`, `use_cov_penalty`, `use_var_penalty` — loss.
- `save_every`, `debug`, `debug_every` — checkpointing / debug dumps.

### Outputs

```
${CONDOR_OUT}/${run_name}/
├── ${run_name}.sub                    # generated submit file
├── <ClusterId>.<ProcId>.{out,err,log} # condor logs
├── checkpoints/                       # rsynced from scratch
└── debug/                             # rsynced from scratch, includes config.json
```
