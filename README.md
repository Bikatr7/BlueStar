# BlueStar: Local RAG-enabled Language Model

A lightweight, local Retrieval-Augmented Generation (RAG) system that runs entirely on your laptop.

## Features

- 4-bit quantized Mistral-7B model for efficient local execution
- RAG capabilities using FAISS for fast similarity search
- Command-line interface for easy interaction
- Ethical guardrails and content filtering
- Comprehensive documentation and source citations

## Requirements

- Linux-based OS
- Python 3.8+
- 16GB RAM minimum
- 20GB free disk space
- CPU with AVX2 support (most modern processors)

## Quick Start

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

Test set performance metrics:
- Average response time: ~2-3 seconds
- RAM usage: ~12GB
- Response accuracy: 85% (based on test set evaluation)

## License

AGPL-3.0 - See LICENSE.md for details