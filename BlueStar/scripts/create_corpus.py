import os
from datasets import load_dataset

def download_wikipedia_corpus(num_articles=1000):
    """Downloads Wikipedia articles and saves them as individual files"""
    corpus_dir = "../data/corpus"
    os.makedirs(corpus_dir, exist_ok=True)
    
    print("Loading Wikipedia dataset...")
    dataset = load_dataset(
        "wikipedia", 
        "20220301.en", 
        split=f"train[:{num_articles}]"
    )
    
    print(f"Processing {len(dataset)} Wikipedia articles...")
    for i, article in enumerate(dataset):
        title = article['title'].replace('/', '_')  ## Clean filename
        content = f"Title: {article['title']}\n\n{article['text']}"
        
        ## Save article
        filename = f"{i:04d}_{title[:50]}.txt"  ## Limit filename length
        filepath = os.path.join(corpus_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if i % 100 == 0:
            print(f"Processed {i} articles...")
    
    print(f"Created corpus with {len(dataset)} articles in {corpus_dir}")

if __name__ == "__main__":
    download_wikipedia_corpus()
