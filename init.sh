#!/usr/bin/env bash
set -euo pipefail

# create activate miniconda env
conda create --name atomizer python=3.10
echo "activating env"
conda activate atomizer

conda install pip
pip install --user -e .
