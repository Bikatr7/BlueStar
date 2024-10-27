#!/bin/bash

## Prompt for Hugging Face token first
echo "Before proceeding, you need to:"
echo "1. Create a Hugging Face account at https://huggingface.co/join"
echo "2. Visit https://huggingface.co/mistralai/Mistral-7B-v0.1"
echo "3. Click 'Access repository' and agree to share your contact information"
echo "4. Get your access token from https://huggingface.co/settings/tokens"
echo
read -p "Enter your Hugging Face token: " HF_TOKEN

## Login to Hugging Face first
echo "Logging in to Hugging Face..."
python3 -c "from huggingface_hub import login; login('$HF_TOKEN')"
if [ $? -ne 0 ]; then
    echo "Failed to login to Hugging Face! Please check your token and try again."
    echo "Make sure you've agreed to share contact information at https://huggingface.co/mistralai/Mistral-7B-v0.1"
    exit 1
fi

## Update and install system dependencies
sudo apt-get update
sudo apt-get install -y git python3 python3-pip build-essential

## Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
    if [ $? -ne 0 ]; then
        echo "Failed to remove existing virtual environment!"
        echo "Please close any programs that might be using it and try again."
        exit 1
    fi
fi

## Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment!"
    exit 1
fi

## Upgrade pip
pip install --upgrade pip

## Install requirements
pip install -r requirements.txt

## Create necessary directories
mkdir -p models/mistral-7b
mkdir -p data/corpus

## Download model if not exists
echo "Checking for existing model..."
if [ -f "models/mistral-7b/config.json" ]; then
    echo "Found existing model files, skipping download..."
else
    echo "Downloading Mistral-7B model..."
    cd models/mistral-7b
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mistralai/Mistral-7B-v0.1', local_dir='.')"
    if [ $? -ne 0 ]; then
        echo "Failed to download model!"
        echo "Please ensure you've agreed to share contact information at https://huggingface.co/mistralai/Mistral-7B-v0.1"
        cd ../..
        exit 1
    fi
    cd ../..
fi

## Create corpus and build retrieval index
echo "Creating corpus..."
python3 scripts/create_corpus.py
if [ $? -ne 0 ]; then
    echo "Failed to create corpus!"
    exit 1
fi

echo "Building retrieval index..."
python3 scripts/build_retrieval.py
if [ $? -ne 0 ]; then
    echo "Failed to build retrieval index!"
    exit 1
fi

## Quantize model
echo "Quantizing model..."
python3 scripts/quantize_model.py
if [ $? -ne 0 ]; then
    echo "Failed to quantize model!"
    exit 1
fi

echo "Setup Complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start the CLI, run: python3 scripts/run_cli.py"
