import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def quantize_model(model_name: str, quantized_model_path: str):
    ## Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    ## Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")

if __name__ == "__main__":
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    QUANTIZED_MODEL_PATH = "../models/quantized-mistral-7b"
    quantize_model(MODEL_NAME, QUANTIZED_MODEL_PATH)
