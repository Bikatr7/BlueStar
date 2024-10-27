#!/bin/bash

# Store the root directory
ROOT_DIR="$(pwd)"

apt install python3.10-venv

## Create virtual environment
python3 -m venv venv
source venv/bin/activate

## Install dependencies first
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

## Now prompt for Hugging Face token
echo "Before proceeding, you need to:"
echo "1. Create a Hugging Face account at https://huggingface.co/join"
echo "2. Visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1"
echo "3. Click 'Access repository' and agree to share your contact information"
echo "4. Get your access token from https://huggingface.co/settings/tokens"
echo
read -p "Enter your Hugging Face token: " HF_TOKEN

## Login to Hugging Face
echo "Logging in to Hugging Face..."
python3 -c "from huggingface_hub import login; login('$HF_TOKEN')"
if [ $? -ne 0 ]; then
    echo "Failed to login to Hugging Face! Please check your token and try again."
    exit 1
fi

## Create all necessary directories
mkdir -p "${ROOT_DIR}/BlueStar/data/corpus"
mkdir -p "${ROOT_DIR}/BlueStar/data/raw"
mkdir -p "${ROOT_DIR}/BlueStar/models"

## Create corpus and build retrieval index
echo "Creating corpus..."
python3 "${ROOT_DIR}/BlueStar/scripts/create_corpus.py"
if [ $? -ne 0 ]; then
    echo "Error creating corpus. Please check the error message above."
    exit 1
fi

echo "Building retrieval index..."
python3 "${ROOT_DIR}/BlueStar/scripts/build_retrieval.py"
if [ $? -ne 0 ]; then
    echo "Error building retrieval index. Please check the error message above."
    exit 1
fi

echo "Setup Complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start the CLI, run: python3 BlueStar/scripts/run_cli.py"
