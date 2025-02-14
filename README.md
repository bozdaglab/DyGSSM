# DyGSSM

# Create and activate environment

```shell script
conda create -p env_name python=3.8.10 -y
conda activate env_name
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install dgl-cu117 -f https://data.dgl.ai/wheels/repo.html
```

Install the python dependencies

```shell script
pip install -r requirements.txt

```