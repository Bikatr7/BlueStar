import click
import os
import sys
import threading
import time
from itertools import cycle

## Add the BlueStar directory to the Python path
utils_dir = os.path.dirname(os.path.abspath(__file__))
bluestar_dir = os.path.dirname(os.path.dirname(utils_dir))
sys.path.insert(0, bluestar_dir)

from BlueStar.utils.retrieval import Retriever
from BlueStar.utils.generation import RAGModel

def spinner_task():
    """Animated spinner to show the model is working"""
    spinner = cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    while True:
        sys.stdout.write(f"\rThinking {next(spinner)} ")
        sys.stdout.flush()
        time.sleep(0.1)

@click.command()
@click.option('--model-path', 
    default=os.path.join(os.path.dirname(__file__), "..", "models", "quantized-mistral-7b"),
    help='Path to the quantized model.')
@click.option('--index-path',
    default=os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index.bin"),
    help='Path to FAISS index.')
@click.option('--corpus-path',
    default=os.path.join(os.path.dirname(__file__), "..", "data", "corpus.pkl"),
    help='Path to the corpus pickle file.')
def main(model_path, index_path, corpus_path):
    try:
        click.echo("Initializing BlueStar...")
        retriever = Retriever(index_path, corpus_path)
        rag = RAGModel(model_path, retriever)
        click.echo("Initialization complete!")
    except Exception as e:
        click.echo(f"Error initializing BlueStar: {e}")
        return

    click.echo("BlueStar RAG CLI. Type 'exit' to quit.")
    while True:
        try:
            query = click.prompt('You', type=str)
            if query.lower() in ['exit', 'quit']:
                break
                
            ## Start spinner in a separate thread
            spinner = threading.Thread(target=spinner_task)
            spinner.daemon = True
            spinner.start()
            
            ## Generate response
            response, sources = rag.generate_response(query)
            
            ## Clear spinner line and print response
            sys.stdout.write('\r' + ' ' * 20 + '\r')  ## Clear spinner
            click.echo(f"BlueStar: {response}")
            if sources:
                click.echo("Sources:")
                for i, doc in enumerate(sources, 1):
                    click.echo(f"{i}. {doc[:200]}...")
                    
        except Exception as e:
            sys.stdout.write('\r' + ' ' * 20 + '\r')  ## Clear spinner
            click.echo(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
