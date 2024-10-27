import os
from datasets import load_dataset

def download_wikipedia_corpus(num_articles=1000):
    """Downloads Wikipedia articles and saves them as individual files"""
    corpus_dir = "../data/corpus"
    os.makedirs(corpus_dir, exist_ok=True)
    
    print("Loading Wikipedia dataset...")
    ## Load a specific subset focused on ML/AI topics
    dataset = load_dataset(
        "wikipedia", 
        "20220301.en", 
        split="train",
        streaming=True  ## Stream to avoid loading entire dataset
    )
    
    ## Filter for relevant articles
    keywords = ['machine learning', 'artificial intelligence', 'deep learning', 
                'neural network', 'computer science', 'data science']
    
    print(f"Filtering for relevant articles...")
    articles_saved = 0
    for article in dataset:
        if articles_saved >= num_articles:
            break
            
        ## Check if article is relevant
        if any(keyword in article['text'].lower() for keyword in keywords):
            title = article['title'].replace('/', '_')
            content = f"Title: {article['title']}\n\n{article['text']}"
            
            filename = f"{articles_saved:04d}_{title[:50]}.txt"
            filepath = os.path.join(corpus_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            articles_saved += 1
            if articles_saved % 10 == 0:
                print(f"Saved {articles_saved} relevant articles...")
    
    print(f"Created corpus with {articles_saved} articles in {corpus_dir}")

if __name__ == "__main__":
    download_wikipedia_corpus()
