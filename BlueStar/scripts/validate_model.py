import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
bluestar_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, bluestar_dir)
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from BlueStar.utils.retrieval import Retriever
from BlueStar.utils.evaluation import ModelEvaluator

def calculate_memory_footprint(model, dtype_size=4):
    """Calculate memory footprint in bytes."""
    return sum(p.numel() * dtype_size for p in model.parameters())

def validate_model(model_path: str, test_set: str, index_path: str, corpus_path: str):
    """Validate the quantized model's performance"""
    
    try:
        print("[INFO] [validate_model.py] Loading model and tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cpu',
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        print("[INFO] [validate_model.py] Successfully loaded model and tokenizer")
        print("[INFO] [validate_model.py] Model size:", calculate_memory_footprint(model) / (1024 * 1024), "MB")
        
        print("[INFO] [validate_model.py] Initializing retriever...")
        retriever = Retriever(index_path, corpus_path)
        
        print("[INFO] [validate_model.py] Setting up evaluator...")
        evaluator = ModelEvaluator(model, tokenizer, retriever)
        
        print("[INFO] [validate_model.py] Running evaluation...")
        results = evaluator.run_full_evaluation(
            test_set,
            os.path.join(os.path.dirname(test_set), "evaluation_results.json")
        )
        
        print("[INFO] [validate_model.py] Evaluation Results:")
        print(json.dumps(results, indent=2))
        
        return results
        
    except Exception as e:
        print(f"[ERROR] [validate_model.py] Validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, "..", "models", "quantized-gpt2-large")
    TEST_SET = os.path.join(script_dir, "..", "data", "test_set.txt")
    INDEX_PATH = os.path.join(script_dir, "..", "data", "faiss_index.bin")
    CORPUS_PATH = os.path.join(script_dir, "..", "data", "corpus.pkl")
    
    validate_model(MODEL_PATH, TEST_SET, INDEX_PATH, CORPUS_PATH)
