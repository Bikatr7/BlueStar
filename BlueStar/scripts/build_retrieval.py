import os
import faiss
from sentence_transformers import SentenceTransformer
import pickle

def load_corpus(corpus_dir: str):
    documents = []
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as f:
                documents.append(f.read())
    return documents

def build_embeddings(documents, model_name: str = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings, index_path: str):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index built and saved to {index_path}")

def save_corpus(documents, corpus_path: str):
    with open(corpus_path, 'wb') as f:
        pickle.dump(documents, f)
    print(f"Corpus saved to {corpus_path}")

if __name__ == "__main__":
    ## Get the absolute path to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ## Define paths relative to the script location
    CORPUS_DIR = os.path.join(script_dir, "..", "data", "corpus")
    INDEX_PATH = os.path.join(script_dir, "..", "data", "faiss_index.bin")
    CORPUS_PATH = os.path.join(script_dir, "..", "data", "corpus.pkl")

    documents = load_corpus(CORPUS_DIR)
    save_corpus(documents, CORPUS_PATH)
    embeddings = build_embeddings(documents)
    build_faiss_index(embeddings, INDEX_PATH)
