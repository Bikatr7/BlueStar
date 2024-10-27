import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

def quantize_model(model_name: str, quantized_model_path: str):
    """Quantize model using a simpler approach"""
    
    print("\nLoading model...")
    loading_bar = tqdm(total=100, desc="Loading model", ncols=100)
    
    try:
        ## Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        loading_bar.update(100)
        loading_bar.close()
    except Exception as e:
        loading_bar.close()
        print(f"Error loading model: {e}")
        return

    print("\nQuantizing model...")
    quantize_bar = tqdm(total=100, desc="Quantizing model", ncols=100)
    
    try:
        ## Move model to CPU to ensure compatibility
        model = model.cpu()
        ## Convert to half precision
        model = model.half()
        quantize_bar.update(100)
        quantize_bar.close()
    except Exception as e:
        quantize_bar.close()
        print(f"Error quantizing model: {e}")
        return
    
    print(f"\nSaving quantized model to {quantized_model_path}...")
    save_bar = tqdm(total=100, desc="Saving model", ncols=100)
    
    try:
        ## Create directory if it doesn't exist
        os.makedirs(quantized_model_path, exist_ok=True)
        
        ## Save configuration first
        model.config.save_pretrained(quantized_model_path)
        ## Save tokenizer
        tokenizer.save_pretrained(quantized_model_path)
        ## Save model weights
        torch.save(model.state_dict(), os.path.join(quantized_model_path, "pytorch_model.bin"))
        
        save_bar.update(100)
        save_bar.close()
        print("\nQuantization complete!")
        
    except Exception as e:
        save_bar.close()
        print(f"Error saving model: {e}")
        return

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_NAME = os.path.join(script_dir, "..", "models", "mistral-7b")
    QUANTIZED_MODEL_PATH = os.path.join(script_dir, "..", "models", "quantized-mistral-7b")
    quantize_model(MODEL_NAME, QUANTIZED_MODEL_PATH)
