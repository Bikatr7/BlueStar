import os
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm
import logging
import sys

## Set up logging with custom format
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s][%(filename)s] %(message)s',
    handlers=[
        logging.FileHandler('build_retrieval.log'),
        logging.StreamHandler()
    ]
)

def load_corpus(corpus_dir: str):
    """Load documents from corpus directory"""
    documents = []
    try:
        files = os.listdir(corpus_dir)
        for filename in tqdm(files, desc="Loading corpus"):
            if filename.endswith(".txt"):
                with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        logging.info(f"Loaded {len(documents)} documents from corpus")
        return documents
    except Exception as e:
        logging.error(f"Error loading corpus: {str(e)}")
        raise

def build_embeddings(documents, model_name: str = 'all-MiniLM-L6-v2'):
    """Build embeddings using sentence transformer"""
    try:
        model = SentenceTransformer(model_name)
        logging.info(f"Building embeddings using {model_name}")
        embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        logging.info(f"Built embeddings with shape {embeddings.shape}")
        return embeddings
    except Exception as e:
        logging.error(f"Error building embeddings: {str(e)}")
        raise

def build_faiss_index(embeddings, index_path: str):
    """Build and save FAISS index"""
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        logging.info(f"FAISS index built and saved to {index_path}")
    except Exception as e:
        logging.error(f"Error building FAISS index: {str(e)}")
        raise

def save_corpus(documents, corpus_path: str):
    """Save corpus to pickle file"""
    try:
        with open(corpus_path, 'wb') as f:
            pickle.dump(documents, f)
        logging.info(f"Corpus saved to {corpus_path}")
    except Exception as e:
        logging.error(f"Error saving corpus: {str(e)}")
        raise

if __name__ == "__main__":
    try:
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
        
        logging.info("Build retrieval process completed successfully")
        
    except Exception as e:
        logging.error(f"Build retrieval process failed: {str(e)}")
        sys.exit(1)
