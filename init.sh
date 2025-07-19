#!/usr/bin/env bash
set -euo pipefail

# create and activate miniconda env
echo "creating a python3.10 env"
conda create --name atomizer python=3.10
echo "activating env"
conda activate atomizer

conda install pip
pip install --user -e .
