import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import json

def calculate_memory_footprint(model, dtype_size=4):
    """Calculate memory footprint in bytes."""
    return sum(p.numel() * dtype_size for p in model.parameters())

def quantize_model(model_name: str, quantized_model_path: str):
    """Quantize model using PyTorch's dynamic quantization for CPU"""
    
    print("[INFO][quantize_model.py] Loading model...")
    loading_bar = tqdm(total=100, desc="Loading model", ncols=100)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        loading_bar.update(100)
        loading_bar.close()
    except Exception as e:
        loading_bar.close()
        print(f"[ERROR][quantize_model.py] Error loading model: {e}")
        return

    print("[INFO][quantize_model.py] Quantizing model...")
    quantize_bar = tqdm(total=100, desc="Quantizing model", ncols=100)
    
    try:
        ## Ensure model is in eval mode
        model.eval()
        
        ## Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  ## Only quantize linear layers
            dtype=torch.qint8
        )
        
        quantize_bar.update(100)
        quantize_bar.close()
    except Exception as e:
        quantize_bar.close()
        print(f"[ERROR][quantize_model.py] Error quantizing model: {e}")
        return
    
    print(f"[INFO][quantize_model.py] Saving quantized model to {quantized_model_path}...")
    save_bar = tqdm(total=100, desc="Saving model", ncols=100)
    
    try:
        os.makedirs(quantized_model_path, exist_ok=True)
        
        ## Save model configuration
        model.config.save_pretrained(quantized_model_path)
        
        ## Save tokenizer
        tokenizer.save_pretrained(quantized_model_path)
        
        ## Manually process and save state_dict, excluding non-tensor items
        state_dict = {}
        for key, value in quantized_model.state_dict().items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.cpu()
            # Skip non-tensor items like 'lm_head.scale' and 'lm_head.zero_point'
        
        ## Save the state_dict using torch.save
        torch.save(state_dict, os.path.join(quantized_model_path, "pytorch_model.bin"))
        
        ## Save additional metadata to a separate file instead of overwriting config.json
        metadata = {
            "model_name": model_name,
            "quantization_config": {
                "dtype": "int8",
                "quantized_layers": "linear",
                "original_size_mb": calculate_memory_footprint(model) / (1024 * 1024),
                "quantized_size_mb": calculate_memory_footprint(quantized_model, dtype_size=1) / (1024 * 1024)
            },
            "model_type": "gpt2",
            "torch_dtype": "int8",
            "transformers_version": "4.44.2"
        }
        
        with open(os.path.join(quantized_model_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        save_bar.update(100)
        save_bar.close()
        print("[SUCCESS][quantize_model.py] Quantization complete!")
        
        ## Calculate and print memory savings
        original_size = calculate_memory_footprint(model) / (1024 * 1024)
        quantized_size = calculate_memory_footprint(quantized_model, dtype_size=1) / (1024 * 1024)
        print(f"[INFO][quantize_model.py] Model size reduced from {original_size:.1f}MB to {quantized_size:.1f}MB")
        print(f"[INFO][quantize_model.py] Compression ratio: {original_size/quantized_size:.1f}x")
        
    except Exception as e:
        save_bar.close()
        print(f"[ERROR][quantize_model.py] Error saving model: {e}")
        return

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_NAME = "gpt2-large"
    QUANTIZED_MODEL_PATH = os.path.join(script_dir, "..", "models", "quantized-gpt2-large")
    quantize_model(MODEL_NAME, QUANTIZED_MODEL_PATH)
