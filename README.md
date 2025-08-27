# DyGSSM
DyGSSM: Multi-view Dynamic Graph Embeddings with State
Space Model Gradient Update

# This repository is our PyTorch implementation of DyGSSM.


## How to run 
### Create and activate environment
```shell script
conda create -p dygssm_env python=3.8.10 -y
conda activate dygssm_env
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install dgl-cu117 -f https://data.dgl.ai/wheels/repo.html
```


```shell script

```
### Install the python dependencies
```shell script
pip install -r requirements.txt

```

### Clone HawkesGNN, ROLAND, and WinGNN code 
1 - Download roland code 
```shell script
git clone git@github.com:snap-stanford/roland.git
git clone git@github.com:oncemoe/hawkesGNN.git
```

2 - Move files from extra_ to their coresponding folders 
```shell script
mv main_roland_call_wingnn.py roland/run
```
3 - Add ROLAND  
```shell script
cd roland
pip install -e .
get_roland_public_data.sh will generate the public dataset folder and download the datasets
```


4 - run 
```shell script
python main.py
```



