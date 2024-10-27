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
