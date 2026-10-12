#!/bin/bash
set -e

echo "Using pip version:"
pip --version

pip install -r ./conda_env/requirements.txt
pip install vcs-versioning
export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9"
export IABN_FORCE_CUDA=1
export NVCC_APPEND_FLAGS="-Xcompiler -Wno-template-body"

git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python setup.py install
cd scripts
pip install -r requirements.txt

cd ../../
rm -rf inplace_abn