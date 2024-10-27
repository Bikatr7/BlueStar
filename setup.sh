#!/bin/bash

# Update and install system dependencies
sudo apt-get update
sudo apt-get install -y git python3 python3-pip build-essential

git clone https://github.com/Bikatr7/BlueStar.git
cd BlueStar

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

mkdir -p models/mistral-7b
cd models/mistral-7b
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mistral-7b', local_dir='.')"

cd ../../

python scripts/build_retrieval.py

echo "Setup Complete. Activate the virtual environment with 'source venv/bin/activate'"
