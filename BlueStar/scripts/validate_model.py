from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def validate_model(model_path: str, test_set: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

    with open(test_set, 'r') as f:
        queries = f.readlines()

    start_time = time.time()
    for query in queries:
        inputs = tokenizer.encode(query, return_tensors='pt')
        outputs = model.generate(inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Query: {query.strip()}\nResponse: {response}\n")
    end_time = time.time()
    print(f"Validation completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    MODEL_PATH = "../models/quantized-mistral-7b"
    TEST_SET = "../data/test_set.txt"
    validate_model(MODEL_PATH, TEST_SET)
