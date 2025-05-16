# DyGSSM
DyGSSM: Multi-view Dynamic Graph Embeddings with State
Space Model Gradient Update

# This repository is our PyTorch implementation of DyGSSM.


## How to run 
# Create and activate environment
```shell script
conda create -p dygssm_env python=3.8.10 -y
conda activate dygssm_env
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install dgl-cu117 -f https://data.dgl.ai/wheels/repo.html
```

Install the python dependencies

```shell script
pip install -r requirements.txt

```

### To run GyGSSM on ROLAND dataset
1 - Download roland code 
```shell script
git@github.com:snap-stanford/roland.git

```
2 - cut and past main_roland_call_wingnn.py into roland/run repo

3 - run 
```shell script
cd roland
pip install -e .
get_roland_public_data.sh will generate the public dataset folder and download the datasets
```


### Acknowledgement
Our source code and data processing are built heavily based on the code of [ROLAND](https://github.com/snap-stanford/roland) and [WinGNN](https://github.com/pursuecong/WinGNN)


