import torch
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import json
import time
import os

class ModelEvaluator:
    def __init__(self, model, tokenizer, retriever=None):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity score for given text"""
        try:
            encodings = self.tokenizer(text, return_tensors="pt")
            max_length = 512
            stride = 128
            seq_len = encodings.input_ids.size(1)
            
            nlls = []
            prev_end_loc = 0
            
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                target_len = end_loc - prev_end_loc
                
                input_ids = encodings.input_ids[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-target_len] = -100
                
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss
                
                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
                
            return torch.exp(torch.stack(nlls).mean()).item()
            
        except Exception as e:
            print(f"[ERROR] [evaluation.py] Error calculating perplexity: {e}")
            return float('inf')
    
    def evaluate_speed(self, text: str, num_runs: int = 3) -> Dict[str, float]:
        """Measure inference speed"""
        try:
            encodings = self.tokenizer(text, return_tensors="pt")
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    self.model.generate(
                        encodings.input_ids,
                        max_length=100,
                        num_return_sequences=1
                    )
                times.append(time.time() - start_time)
            
            return {
                "avg_inference_time": np.mean(times),
                "std_inference_time": np.std(times)
            }
            
        except Exception as e:
            print(f"[ERROR] [evaluation.py] Error evaluating speed: {e}")
            return {"avg_inference_time": float('inf'), "std_inference_time": float('inf')}
    
    def evaluate_memory(self) -> Dict[str, float]:
        """Measure memory usage"""
        try:
            memory_bytes = self.model.get_memory_footprint()
            return {
                "model_size_mb": memory_bytes / (1024 * 1024),
                "model_size_gb": memory_bytes / (1024 * 1024 * 1024)
            }
        except Exception as e:
            print(f"[ERROR] [evaluation.py] Error evaluating memory: {e}")
            return {"model_size_mb": float('inf'), "model_size_gb": float('inf')}
    
    def evaluate_retrieval(self, queries: List[str]) -> Dict[str, float]:
        """Evaluate retrieval performance if retriever is available"""
        if not self.retriever:
            return {}
            
        try:
            retrieval_times = []
            for query in queries:
                start_time = time.time()
                self.retriever.retrieve(query)
                retrieval_times.append(time.time() - start_time)
                
            return {
                "avg_retrieval_time": np.mean(retrieval_times),
                "std_retrieval_time": np.std(retrieval_times)
            }
        except Exception as e:
            print(f"[ERROR] [evaluation.py] Error evaluating retrieval: {e}")
            return {}
    
    def run_full_evaluation(self, test_set_path: str, output_path: str = None):
        """Run comprehensive evaluation and save results"""
        results = {
            "perplexity": [],
            "speed": {"inference_times": []},
            "memory": self.evaluate_memory(),
            "retrieval": {}
        }
        
        try:
            with open(test_set_path, 'r', encoding='utf-8') as f:
                test_queries = [line.strip() for line in f if line.strip()]
            
            for query in tqdm(test_queries, desc="Evaluating queries"):
                results["perplexity"].append(self.calculate_perplexity(query))
                speed_metrics = self.evaluate_speed(query)
                results["speed"]["inference_times"].append(speed_metrics["avg_inference_time"])
            
            results["perplexity_avg"] = np.mean(results["perplexity"])
            results["perplexity_std"] = np.std(results["perplexity"])
            results["speed"]["avg_inference_time"] = np.mean(results["speed"]["inference_times"])
            results["speed"]["std_inference_time"] = np.std(results["speed"]["inference_times"])
            
            if self.retriever:
                results["retrieval"] = self.evaluate_retrieval(test_queries)

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            print(f"[ERROR] [evaluation.py] Error in full evaluation: {e}")
            return results

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, "..", "models", "quantized-gpt2-large")
    TEST_SET = os.path.join(script_dir, "..", "data", "test_set.txt")
    RESULTS_PATH = os.path.join(script_dir, "..", "data", "evaluation_results.json")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    evaluator = ModelEvaluator(model, tokenizer)
    
    results = evaluator.run_full_evaluation(TEST_SET, RESULTS_PATH)
    print("\n[INFO] [evaluation.py] Evaluation Results:")
    print(json.dumps(results, indent=2))
