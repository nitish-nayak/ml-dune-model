#!/bin/bash
# build_env.sh
export CUDA=cu128
export TORCH_REL="2.10.0"
export TORCH_RELL="2.10"
export WARPCONV_REL="1.7.3"
export TORCH_CUDA_ARCH_LIST="8.9"
export UV_LINK_MODE=copy

USER=mvicenzi
WARP_WHEEL="https://github.com/NVlabs/WarpConvNet/releases/download/v${WARPCONV_REL}/warpconvnet-${WARPCONV_REL}+torch${TORCH_RELL}${CUDA}-cp311-cp311-linux_x86_64.whl"

echo "Sourcing uv environment!"
source /gpfs01/lbne/users/fm/${USER}/uvenv/bin/activate

echo "Installing pytorch..."
/lbne/u/${USER}/.local/bin/uv pip install torch==${TORCH_REL} torchvision --index-url https://download.pytorch.org/whl/${CUDA}
/lbne/u/${USER}/.local/bin/uv pip install build ninja

echo "Installing torch-scatter.."
/lbne/u/${USER}/.local/bin/uv pip install torch-scatter --find-links https://data.pyg.org/whl/torch-${TORCH_REL}+${CUDA}.html

echo "Installing warpconvnet..."
wget ${WARP_WHEEL}
/lbne/u/${USER}/.local/bin/uv pip install ${WARP_WHEEL}

## if you wish to build warpconvnet from source using a local area
## you will need to do so on GPU by submitting a "build job"
## basically: submit this script, commenting the previous lines and 
## uncommenting the ones below:

#echo "Installing local warpconvnet..."
#/lbne/u/${USER}/.local/bin/uv pip install -e /path/to/local/WarpConvNet --no-cache-dir --no-build-isolation

echo "Installing other run dependencies..."
/lbne/u/${USER}/.local/bin/uv pip install fire h5py

## There is no pre-built wheel for flash-attention for torch2.10+cu128
## this needs to be build from source on GPU by submitting a "build job"
## basically: submit this script, commenting the previous lines and 
## uncommenting the ones below (note: it will take ~1.5 hours)

#echo "Installing flash_attn..."
#/lbne/u/${USER}/.local/bin/uv pip install wheel setuptools psutil
#MAX_JOBS=1 /lbne/u/${USER}/.local/bin/uv pip install flash-attn --no-build-isolation

echo "Done!"
