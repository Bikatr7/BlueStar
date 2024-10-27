#!/bin/bash

# Store the root directory
ROOT_DIR="$(pwd)"

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

## Check if virtual environment exists and is up to date
RECREATE_VENV=0
if [ -d "venv" ]; then
    echo "Found existing virtual environment, checking if it's up to date..."
    ./venv/bin/pip check
    if [ $? -ne 0 ]; then
        echo "Virtual environment is outdated, recreating..."
        RECREATE_VENV=1
    else
        ./venv/bin/pip install -r requirements.txt --dry-run
        if [ $? -ne 0 ]; then
            echo "Virtual environment is missing some packages, recreating..."
            RECREATE_VENV=1
        else
            echo "Virtual environment is up to date, skipping recreation..."
        fi
    fi
else
    RECREATE_VENV=1
fi

## Create/Recreate virtual environment if needed
if [ $RECREATE_VENV -eq 1 ]; then
    if [ -d "venv" ]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
        if [ $? -ne 0 ]; then
            echo "Failed to remove existing virtual environment!"
            echo "Please close any programs that might be using it and try again."
            exit 1
        fi
    fi
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment!"
        exit 1
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

## Create necessary directories
mkdir -p "${ROOT_DIR}/BlueStar/models/mistral-7b"
mkdir -p "${ROOT_DIR}/BlueStar/models/quantized-mistral-7b"
mkdir -p "${ROOT_DIR}/BlueStar/data/corpus"

## Check for model files
echo "Checking for model files..."
MODEL_FILES_MISSING=0
for file in config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors tokenizer.model tokenizer.json; do
    if [ ! -f "${ROOT_DIR}/BlueStar/models/mistral-7b/$file" ]; then
        MODEL_FILES_MISSING=1
        break
    fi
done

if [ $MODEL_FILES_MISSING -eq 1 ]; then
    echo "Some model files are missing. Downloading Mistral-7B..."
    cd "${ROOT_DIR}/BlueStar/models/mistral-7b"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mistralai/Mistral-7B-v0.1', local_dir='.')"
    if [ $? -ne 0 ]; then
        echo "Failed to download model!"
        echo "If this keeps happening, try:"
        echo "1. Ensure you've agreed to share contact information at https://huggingface.co/mistralai/Mistral-7B-v0.1"
        echo "2. Delete the ${ROOT_DIR}/BlueStar/models/mistral-7b directory and run setup again"
        cd "${ROOT_DIR}"
        exit 1
    fi
    cd "${ROOT_DIR}"
else
    echo "Found existing model files, skipping download..."
fi

# Create corpus and build retrieval index only if needed
echo "Checking for existing corpus..."
if ls "${ROOT_DIR}/BlueStar/data/corpus/"*.txt >/dev/null 2>&1; then
    echo "Found existing corpus files, skipping corpus creation..."
else
    echo "Creating corpus..."
    python3 "${ROOT_DIR}/BlueStar/scripts/create_corpus.py"
    if [ $? -ne 0 ]; then
        echo "Failed to create corpus!"
        exit 1
    fi
fi

# Build retrieval index only if needed
if [ -f "${ROOT_DIR}/BlueStar/data/faiss_index.bin" ]; then
    echo "Found existing retrieval index, skipping build..."
else
    echo "Building retrieval index..."
    python3 "${ROOT_DIR}/BlueStar/scripts/build_retrieval.py"
    if [ $? -ne 0 ]; then
        echo "Failed to build retrieval index!"
        exit 1
    fi
fi

## Quantize model
echo "Quantizing model..."
python3 "${ROOT_DIR}/BlueStar/scripts/quantize_model.py"
if [ $? -ne 0 ]; then
    echo "Failed to quantize model!"
    exit 1
fi

echo "Setup Complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start the CLI, run: python3 BlueStar/scripts/run_cli.py"
