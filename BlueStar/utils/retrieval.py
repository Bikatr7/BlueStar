import faiss
import pickle
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path: str, corpus_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.index = faiss.read_index(index_path)
        with open(corpus_path, 'rb') as f:
            self.corpus = pickle.load(f)
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.corpus[idx] for idx in indices[0]]
        return results
