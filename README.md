# BlueStar: Local RAG-enabled Language Model

A lightweight, local Retrieval-Augmented Generation (RAG) system powered by Mistral-7B-Instruct-v0.1, running entirely on CPU.

## Features

- 4-bit quantized Mistral-7B-Instruct model for efficient local execution
- RAG capabilities using FAISS for fast similarity search
- Interactive command-line interface
- Performance monitoring and resource usage tracking
- Ethical guardrails and content filtering
- Query refinement for better responses
- Comprehensive source citations

## Requirements

- Linux-based OS
- Python 3.8+
- 16GB RAM minimum
- 20GB free disk space
- Hugging Face account with access token

## Pre-Installation Steps

1. Create a Hugging Face account at https://huggingface.co/join
2. Visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 and accept terms
3. Get your access token from https://huggingface.co/settings/tokens

## Installation

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

3. When prompted, enter your Hugging Face token.

## Usage

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Run the CLI:
```bash
python3 BlueStar/scripts/run_cli.py
```

## Architecture

The system consists of four main components:

1. **Model Layer**: 4-bit quantized Mistral-7B-Instruct model using BitsAndBytes
2. **Retrieval Layer**: FAISS-based similarity search with sentence transformers
3. **RAG Integration**: Combines retrieved context with model for enhanced responses
4. **Interface Layer**: CLI with performance monitoring and user feedback

## Performance

TBD I'm putting these as filler as I hope that's how it performs lol.

- Average response time: 10-30 seconds (first response), 5-15 seconds (subsequent)
- Memory usage: ~8GB RAM
- Storage: ~15GB (model), ~3GB (environment)

## Features in Detail

### Retrieval-Augmented Generation
- Uses FAISS for efficient similarity search
- Retrieves relevant context from local document corpus
- Enhances responses with source citations

### Ethical Guardrails
- Content filtering for inappropriate topics
- Query refinement for clarity
- Source attribution

### Performance Monitoring
- Real-time CPU usage tracking
- Memory consumption monitoring
- Response time measurements

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Close other applications
   - Ensure at least 16GB RAM available
   - Check no other large models are running

2. **Slow Responses**:
   - First response is always slower due to model loading
   - Check CPU usage
   - Verify no other intensive processes running

3. **Installation Issues**:
   - Verify Hugging Face token is correct
   - Ensure all dependencies are installed
   - Check disk space availability

## License

AGPL-3.0 - See LICENSE.md for details

## Acknowledgments

- Mistral AI for the base model
- Hugging Face for model hosting and libraries
- FAISS team for the similarity search implementation
