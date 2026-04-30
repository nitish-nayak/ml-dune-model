#!/bin/bash
#
# DINO training submission to SDCC GPU pool.
#
# Usage:
#   bash gridutils/submit.sh <path/to/run_config.json>
#
# The JSON config is the single source of truth for the run.  The submitter
# extracts run_name from it, lays out ${CONDOR_OUT}/${run_name}/, generates
# the .sub file there, and submits.  Everything else (hyperparameters, paths,
# debug flags) is read from the JSON by dino.train_dino.from_config on the
# worker node.

set -euo pipefail

# ---- User-overridable env --------------------------------------------------

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


# ---- Args / validation -----------------------------------------------------
if [ $# -ne 1 ]; then
  echo "usage: $0 <run_config.json>" >&2
  exit 2
fi

config=$1
if [ ! -f "$config" ]; then
  echo "ERROR: config file not found: $config" >&2
  exit 1
fi
config=$(cd "$(dirname "$config")" && pwd)/$(basename "$config")  # absolute

# Pull run_name out of the JSON configuration
# This is to make sure things are consistent
run_name=$(jq -r '.run_name' "$config")

if [ -z "$run_name" ] || [ "$run_name" = "None" ]; then
  echo "ERROR: 'run_name' in $config is empty; set a run_name and retry." >&2
  exit 1
fi

out_dir="${CONDOR_OUT}/${run_name}"

if [ -d "$out_dir" ]; then
  echo "ERROR: ${out_dir} already exists." >&2
  echo "       Choose a new run_name or delete the directory and retry." >&2
  exit 1
fi

# ---- Layout ----------------------------------------------------------------
echo "Creating run directory: ${out_dir}"
mkdir -p "$out_dir"

subfile="${out_dir}/${run_name}.sub"
cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/trainjob.sh
arguments               = ${REPODIR} ${PYENV} ${config} ${out_dir} ${CACHE_DIR} ${run_name}
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId)"
output                  = ${out_dir}/\$(ClusterId).\$(ProcId).out
error                   = ${out_dir}/\$(ClusterId).\$(ProcId).err
log                     = ${out_dir}/\$(ClusterId).\$(ProcId).log
getenv                  = False
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
request_gpus            = ${REQUEST_GPUS}
Requirements            = ${GPU_REQUIREMENTS}
should_transfer_files   = NO
queue 1
EOF

echo "Submitting ${subfile}"
condor_submit "$subfile"
