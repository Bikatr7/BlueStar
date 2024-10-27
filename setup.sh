#!/bin/bash

# Store the root directory
ROOT_DIR="$(pwd)"

apt install python3.10-venv

## Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists, skipping creation..."
fi

source venv/bin/activate

## Install dependencies first
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

## Create all necessary directories
mkdir -p "${ROOT_DIR}/BlueStar/data/corpus"
mkdir -p "${ROOT_DIR}/BlueStar/models"

## Download base model
echo "Downloading base model..."
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('gpt2-large', local_files_only=False); AutoTokenizer.from_pretrained('gpt2-large', local_files_only=False)"

## Quantize model
echo "Quantizing model..."
python3 "${ROOT_DIR}/BlueStar/scripts/quantize_model.py"
if [ $? -ne 0 ]; then
    echo "Error quantizing model. Please check the error message above."
    exit 1
fi

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

## Run model validation
echo "Validating model..."
python3 "${ROOT_DIR}/BlueStar/scripts/validate_model.py"
if [ $? -ne 0 ]; then
    echo "Error validating model. Please check the error message above."
    exit 1
fi

echo "Setup Complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start the CLI, run: python3 BlueStar/scripts/run_cli.py"
