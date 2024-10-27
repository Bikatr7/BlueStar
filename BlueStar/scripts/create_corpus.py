import os
from datasets import load_dataset
import logging
from tqdm import tqdm
import sys

## Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_corpus.log'),
        logging.StreamHandler()
    ]
)

def download_wikipedia_corpus(num_articles=1000):
    """Downloads Wikipedia articles and saves them as individual files"""
    try:
        ## Get the absolute path to the BlueStar/data/corpus directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_dir = os.path.join(script_dir, "..", "data", "corpus")
        corpus_dir = os.path.abspath(corpus_dir)
        
        os.makedirs(corpus_dir, exist_ok=True)
        print(f"[INFO][create_corpus.py] Using corpus directory: {corpus_dir}")
        
        print("[INFO][create_corpus.py] Loading Wikipedia dataset...")
        dataset = load_dataset(
            "wikipedia", 
            "20220301.en", 
            split="train",
            streaming=True
        )
        
        ## Filter for relevant articles
        keywords = ['machine learning', 'artificial intelligence', 'deep learning', 
                   'neural network', 'computer science', 'data science']
        
        print(f"[INFO][create_corpus.py] Filtering for relevant articles containing keywords: {keywords}")
        articles_saved = 0
        
        progress_bar = tqdm(total=num_articles, desc="Saving articles")
        
        for article in dataset:
            if articles_saved >= num_articles:
                break
                
            ## Check if article is relevant
            if any(keyword in article['text'].lower() for keyword in keywords):
                try:
                    ## Clean the title to be filesystem-safe
                    title = "".join(c for c in article['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    content = f"Title: {article['title']}\n\n{article['text']}"
                    
                    filename = f"{articles_saved:04d}_{title[:50]}.txt"
                    filepath = os.path.join(corpus_dir, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    articles_saved += 1
                    progress_bar.update(1)
                    
                except Exception as e:
                    print(f"[WARNING][create_corpus.py] Error saving article {article['title']}: {str(e)}")
                    continue
        
        progress_bar.close()
        print(f"[INFO][create_corpus.py] Created corpus with {articles_saved} articles in {corpus_dir}")
        
    except Exception as e:
        print(f"[ERROR][create_corpus.py] Error creating corpus: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        download_wikipedia_corpus()
    except Exception as e:
        print(f"[ERROR][create_corpus.py] Corpus creation failed: {str(e)}")
        sys.exit(1)
