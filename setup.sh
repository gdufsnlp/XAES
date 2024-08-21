#!/usr/bin/bash
set -ex

venv=".venv"
if [ ! -d ${venv} ]; then
  python3.7 -m venv ${venv}
fi
source ${venv}/bin/activate

pip install -r requirements.txt