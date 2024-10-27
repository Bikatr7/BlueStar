# BlueStar: Local RAG-enabled Language Model

A lightweight, local Retrieval-Augmented Generation (RAG) system that runs entirely on your laptop.

## Features

- 4-bit quantized Mistral-7B model for efficient local execution
- RAG capabilities using FAISS for fast similarity search
- Command-line interface for easy interaction
- Ethical guardrails and content filtering
- Comprehensive documentation and source citations

## Requirements

- Linux-based OS or Windows 10/11 (yet to be tested on either yet)
- Python 3.8+
- 16GB RAM minimum
- 20GB free disk space
- CPU with AVX2 support (most modern processors)
- Hugging Face account with access token

## Pre-Installation Steps

1. Create a Hugging Face account at https://huggingface.co/join
2. Visit https://huggingface.co/settings/tokens to create an access token
3. Accept the Mistral-7B model terms at https://huggingface.co/mistralai/Mistral-7B-v0.1

## Quick Start

### Windows Installation

1. Clone the repository:
```bash
git clone https://github.com/Bikatr7/BlueStar.git
cd BlueStar
```

2. Run the setup script:
```bash
setup.bat
```
When prompted, enter your Hugging Face token.

3. Run the CLI:
```bash
venv\Scripts\activate.bat
python scripts\run_cli.py
```

### Linux Installation

1. Clone the repository:
```bash
git clone https://github.com/Bikatr7/BlueStar.git
cd BlueStar
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```
When prompted, enter your Hugging Face token.

3. Run the CLI:
```bash
source venv/bin/activate
python scripts/run_cli.py
```

## Architecture

The system consists of three main components:

1. **Quantized Language Model**: A 4-bit quantized Mistral-7B model for efficient local execution
2. **Retrieval System**: FAISS-based similarity search with sentence transformers
3. **RAG Integration**: Combines retrieved context with the language model for enhanced responses

## Validation Results

TBD

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Make sure you have:
   - Created a Hugging Face account
   - Generated an access token
   - Accepted the model terms of use
   - Entered the correct token during setup

2. **Out of Memory**: The system requires at least 16GB RAM. Close other applications if needed.

## License

AGPL-3.0 - See LICENSE.md for details