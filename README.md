# BlueStar: Local RAG-enabled Language Model

A lightweight, local Retrieval-Augmented Generation (RAG) system powered by GPT-2 Large (774M parameters), running entirely on CPU.

## Note to BlueStaq Reviewers

The model does not perform that "well", I haven't had the time to strictly optimize the thing, merely just the day to get it done and have something presentable. Frankly, I tried, but I am unable to deliver such a product in 5 days while going to school full time and working an internship.

I'd be more than happy to discuss it, and possibly continue along. But I have other avenues to pursue for this summer and hope to try again perhaps another time.

Thank you for your time.

## Features

- **Local Execution:** Runs entirely on your local machine without the need for cloud resources.
- **CPU-optimized:** Utilizes PyTorch's dynamic quantization (8-bit) to reduce model size and improve inference speed.
- **RAG Capabilities:** Enhances responses using FAISS for fast similarity search over a local document corpus.
- **Interactive Command-Line Interface:** Engage with the model seamlessly via a user-friendly CLI.
- **Performance Monitoring:** Tracks CPU and RAM usage, as well as inference times during operation.
- **Ethical Guardrails:** Implements content filtering to restrict inappropriate topics.
- **Query Refinement:** Improves user queries for clearer and more accurate responses.
- **Comprehensive Documentation:** Detailed setup instructions, usage guides, and troubleshooting tips.

## Model Optimization

BlueStar leverages PyTorch's dynamic quantization to optimize the GPT-2 Large model for efficient CPU inference:

- **Original Model Size:** ~3GB (float32)
- **Quantized Model Size:** ~750MB (int8)
- **Compression Ratio:** Approximately 4.0x reduction in model size
- **Average Inference Time:** 5-30 seconds per query (varies with query length, complexity, and system load)
- **RAM Usage:** ~1GB during operation
- **CPU Usage:** ~10% idle, 90% busy

### **Quantization Process:**

1. **Load the Model:** Initializes the GPT-2 Large model in float32 precision.
2. **Apply Dynamic Quantization:** Quantizes only the linear layers (`torch.nn.Linear`) to 8-bit integers using `torch.quantization.quantize_dynamic`.
3. **Save Quantized Model:** Saves the quantized model's state dictionary as `pytorch_model.bin` and preserves the model configuration.
4. **Metadata Storage:** Stores additional quantization details in `metadata.json` to maintain compatibility and track optimization metrics.

## Requirements

- **Python:** 3.10+
- **RAM:** 8GB minimum
- **Disk Space:** 40-50GB free (i'll clean this up later hopefully (i did not))
- **Operating System:** Windows or Linux (Ubuntu-based distros supported out of the box)

## Installation

### **Windows**

1. **Clone the Repository:**
    ```batch
    git clone https://github.com/Bikatr7/BlueStar.git
    cd BlueStar
    ```

2. **Run the Setup Script:**
    ```batch
    setup.bat
    ```

### **Linux**

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Bikatr7/BlueStar.git
    cd BlueStar
    ```

2. **Run the Setup Script:**
    ```bash
    sh setup.sh
    ```

## Setup Process

The setup script (`setup.bat` for Windows and `setup.sh` for Linux) automates the following steps:

1. **Virtual Environment Creation:** Sets up a Python virtual environment to isolate project dependencies.
2. **Dependency Installation:** Installs all required Python packages listed in `requirements.txt`.
3. **Model Download and Quantization:**
    - Downloads the GPT-2 Large model using Hugging Face Transformers.
    - Applies dynamic quantization to reduce the model size and improve CPU inference speed.
4. **Corpus Creation and Retrieval Indexing:**
    - Fetches and filters Wikipedia articles relevant to predefined keywords.
    - Builds FAISS indices over the document corpus for efficient retrieval.
5. **Model Validation:** Validates the quantized model's performance against a test set.

## Usage

### **Running the Command-Line Interface**

1. **Activate the Virtual Environment:**

    - **Windows:**
        ```batch
        venv\Scripts\activate.bat
        ```

    - **Linux:**
        ```bash
        source venv/bin/activate
        ```

2. **Launch the CLI:**
    ```bash
    python BlueStar\scripts\run_cli.py
    ```

3. **Interact with the Model:**
    - Type your queries when prompted.
    - Type `exit` or `quit` to terminate the session.

### **Example Interaction:**
```
Initializing BlueStar...
BlueStar RAG CLI. Type 'exit' to quit.
You: What is machine learning?
Thinking ⠋ 
BlueStar: Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience without being explicitly programmed. It involves the use of data to train models that can make predictions or decisions based on new, unseen data.

Sources:
1. Machine Learning is a field of computer science that gives computers the ability to learn without being explicitly programmed. It is closely related to computational statistics.
```

## Architecture

BlueStar's architecture is divided into four core components:

1. **Model Layer:** GPT-2 Large model optimized with PyTorch's dynamic quantization.
2. **Retrieval Layer:** FAISS-based similarity search utilizing sentence transformers for efficient document retrieval.
3. **RAG Integration:** Combines retrieved documents with the model to generate informed and contextually relevant responses.
4. **Interface Layer:** Command-Line Interface (CLI) that facilitates user interaction, performance monitoring, and content filtering.

## Performance

### Not tested in detail (not good)

## Features in Detail

### **Retrieval-Augmented Generation (RAG)**

- **FAISS Integration:** Utilizes FAISS for efficient similarity searches over the document corpus.
- **Contextual Responses:** Enhances generated responses with relevant information retrieved from the local corpus.
- **Source Citations:** Provides references to the sources used in generating responses.

### **Ethical Guardrails**

- **Content Filtering:** Restricts topics related to violence, illegal activities, hate speech, weapons, drugs, and more.
- **Command Restrictions:** Blocks harmful commands like `sudo`, `rm -rf`, `format`, and `del`.
- **User Feedback:** Informs users when queries fall outside allowed topics and prompts for more specific inputs when necessary.

### **Performance Monitoring**

- **Resource Tracking:** Monitors real-time CPU and RAM usage during interactions.
- **Response Time Logging:** Measures and displays the time taken to generate each response.
- **Resource Metrics:** Outputs metrics related to model size, inference speed, and retrieval performance.

## Troubleshooting

### **Common Issues**

1. **Out of Memory:**
    - **Solution:**
        - Close other applications to free up RAM.
        - Use shorter queries to reduce processing load.
        - Ensure the system meets the minimum RAM requirements.

2. **Slow Responses:**
    - **Solution:**
        - First-time queries may be slower due to model loading.
        - Check CPU usage to ensure no other intensive processes are running.
        - Restart the CLI to refresh the model loading process.

3. **Installation Errors:**
    - **Solution:**
        - Verify that Python 3.10+ is installed.
        - Check for sufficient disk space.
        - Ensure all dependencies are correctly installed by re-running the setup script.

4. **Model Loading Errors:**
    - **Solution:**
        - Ensure that the `quantized-gpt2-large` directory contains `pytorch_model.bin` and `config.json`.
        - Re-run the quantization process if necessary.
        - Verify that the environment variables and paths are correctly set.

## License

AGPL-3.0 - See `LICENSE.md` for details

## Acknowledgments

- **OpenAI:** For the GPT-2 model.
- **Hugging Face:** For model hosting and the Transformers library.
- **FAISS Team:** For the efficient similarity search implementation.
- **SentenceTransformers:** For embedding generation.

## Future Directions

### If Given More Time

- **Expanded Corpus:** Integrating additional datasets to enhance the model's knowledge base.
- **Better Generation** Model currently performs poorly. This can be improved.
- **Configurable Corpus** Currently just fetches data from hardcoded key words.
- **Overal Optimization** Frankly, a lot can be optimized. In the few short days I was given I was mostly on break, working, or school. I'd probably rework it from the ground up given another chance.
- **GPU Support** Optinal GPU support would be great.

## Contact

For any questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/Bikatr7/BlueStar).
