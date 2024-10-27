import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List

class Retriever:
    def __init__(self, index_path: str, corpus_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            self.index = faiss.read_index(index_path)
            with open(corpus_path, 'rb') as f:
                self.corpus = pickle.load(f)
            self.model = SentenceTransformer(model_name)
            print(f"[INFO] [retrieval.py] Retriever initialized with {len(self.corpus)} documents")
        except Exception as e:
            print(f"[ERROR] [retrieval.py] Failed to initialize retriever: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            distances, indices = self.index.search(query_embedding, top_k)
            
            ## Log retrieval metrics
            print(f"[DEBUG] [retrieval.py] Query: {query}")
            print(f"[DEBUG] [retrieval.py] Top {top_k} distances: {distances[0]}")
            
            results = [self.corpus[idx] for idx in indices[0]]
            return results
        except Exception as e:
            print(f"[ERROR] [retrieval.py] Retrieval failed for query '{query}': {str(e)}")
            return []
