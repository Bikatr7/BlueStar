# BlueStar: Local RAG-enabled Language Model

A lightweight, local Retrieval-Augmented Generation (RAG) system powered by GPT-2 Large, running entirely on CPU.

## Features

- Local execution with GPT-2 Large (774M parameters)
- RAG capabilities using FAISS for fast similarity search
- Interactive command-line interface
- Performance monitoring and resource usage tracking
- Ethical guardrails and content filtering
- Query refinement for better responses
- Comprehensive source citations

## Requirements

- Python 3.10+
- 8GB RAM minimum
- 10GB free disk space
- Windows or Linux OS (Ubuntu-based distros supported)

## Installation

### Windows
1. Clone the repository:
```batch
git clone https://github.com/Bikatr7/BlueStar.git
cd BlueStar
```

2. Run the setup script:
```batch
setup.bat
```

### Linux
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

## Usage

### Windows
1. Activate the virtual environment:
```batch
venv\Scripts\activate.bat
```

2. Run the CLI:
```batch
python BlueStar\scripts\run_cli.py
```

### Linux
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

1. **Model Layer**: GPT-2 Large with CPU optimizations
2. **Retrieval Layer**: FAISS-based similarity search with sentence transformers
3. **RAG Integration**: Combines retrieved context with model for enhanced responses
4. **Interface Layer**: CLI with performance monitoring and user feedback

## Performance

- Average response time: 2-5 seconds
- Memory usage: ~4GB RAM
- Storage: ~5GB (model + environment)

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
   - Ensure at least 8GB RAM available
   - Check no other large processes are running

2. **Slow Responses**:
   - First response is slower due to model loading
   - Check CPU usage
   - Verify no other intensive processes running

3. **Installation Issues**:
   - Ensure Python 3.10+ is installed
   - Check disk space availability
   - Verify all dependencies are installed

## License

AGPL-3.0 - See LICENSE.md for details

## Acknowledgments

- OpenAI for the GPT-2 model
- Hugging Face for model hosting and libraries
- FAISS team for the similarity search implementation
