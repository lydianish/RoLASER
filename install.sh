#!/bin/bash

set -xe

# Install PyTorch 1.10.1
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html

# Install Fairseq and dependencies
git clone https://github.com/lydianish/fairseq.git
cd fairseq
git checkout rolaser
pip install -e .
python setup.py build_ext --inplace
pip install numpy==1.21.6 fairscale==0.4.6
export FAIRSEQ=`pwd` # required environment variable

# Install LASER and dependencies
cd .. # back to RoLASER directory
git clone https://github.com/lydianish/LASER.git
cd LASER
git checkout rolaser
export LASER=`pwd` # required environment variable
bash ./nllb/download_models.sh swh_Latn 
bash ./install_external_tools.sh
pip install faiss-gpu JapaneseTokenizer jieba transliterate tabulate

# Install other RoLASER dependencies
cd .. # back to RoLASER directory
pip install transformers pandas tensorboardX