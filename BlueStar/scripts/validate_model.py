import time
import json
import os
import sys

## Add the BlueStar directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
bluestar_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, bluestar_dir)

from BlueStar.utils.generation import RAGModel
from BlueStar.utils.retrieval import Retriever

def validate_model(model_path: str, test_set: str, index_path: str, corpus_path: str):
    ## Initialize RAG model
    retriever = Retriever(index_path, corpus_path)
    rag = RAGModel(model_path, retriever)
    
    ##   Load test queries
    with open(test_set, 'r') as f:
        queries = [line.strip() for line in f.readlines() if line.strip()]
    
    results = {
        "total_queries": len(queries),
        "total_time": 0,
        "responses": []
    }
    
    for query in queries:
        start_time = time.time()
        response, sources = rag.generate_response(query)
        end_time = time.time()
        
        query_time = end_time - start_time
        results["total_time"] += query_time
        
        results["responses"].append({
            "query": query,
            "response": response,
            "time": query_time,
            "sources_count": len(sources)
        })
    
    ## Calculate metrics
    results["average_time"] = results["total_time"] / len(queries)
    
    ## Save results
    with open("../data/validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Validation completed. Average response time: {results['average_time']:.2f} seconds")
    return results

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, "..", "models", "quantized-mistral-7b")
    TEST_SET = os.path.join(script_dir, "..", "data", "test_set.txt")
    INDEX_PATH = os.path.join(script_dir, "..", "data", "faiss_index.bin")
    CORPUS_PATH = os.path.join(script_dir, "..", "data", "corpus.pkl")
    validate_model(MODEL_PATH, TEST_SET, INDEX_PATH, CORPUS_PATH)
