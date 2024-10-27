import click
import os
import sys
import threading
import time
from itertools import cycle
from BlueStar.utils.metrics import monitor_resources

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
@click.option('--index-path',
    default=os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index.bin"),
    help='Path to FAISS index.')
@click.option('--corpus-path',
    default=os.path.join(os.path.dirname(__file__), "..", "data", "corpus.pkl"),
    help='Path to the corpus pickle file.')
@click.option('--device',
    default='auto',
    help='Device to use for model inference. Options: "auto", "cpu", or "cuda".')
def main(index_path, corpus_path, device):
    try:
        click.echo("Initializing BlueStar...")
        retriever = Retriever(index_path, corpus_path)
        rag = RAGModel(None, retriever, device)
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
            
            ## Check if topic is allowed
            if not rag.is_allowed_topic(query):
                click.echo("I apologize, but I cannot assist with that topic due to ethical constraints.")
                continue
            
            ## Refine query if needed
            refined_query = rag.refine_query(query)
            if refined_query != query:
                click.echo(f"Refining query: {refined_query}")
                query = refined_query
                
            spinner = threading.Thread(target=spinner_task)
            spinner.daemon = True
            spinner.start()
            
            start_time = time.time()
            cpu_start, ram_start = monitor_resources()
            
            ## Generate response
            try:
                response, sources = rag.generate_response(query)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    click.echo("Error: Out of memory. Please try a shorter query or free up system resources.")
                    continue
                else:
                    click.echo(f"An error occurred during generation: {str(e)}")
                    continue
            
            ## Get resource usage
            cpu_end, ram_end = monitor_resources()
            end_time = time.time()
            
            ## Clear spinner line and print response
            sys.stdout.write('\r' + ' ' * 20 + '\r')
            click.echo(f"BlueStar: {response}")
            if sources:
                click.echo("\nSources:")
                for i, doc in enumerate(sources, 1):
                    click.echo(f"{i}. {doc[:200]}...")
            
            ## Print performance metrics
            click.echo(f"\nPerformance Metrics:")
            click.echo(f"Response Time: {end_time - start_time:.2f}s")
            click.echo(f"CPU Usage: {cpu_end - cpu_start:.1f}%")
            click.echo(f"RAM Usage: {ram_end - ram_start:.1f}%")
                    
        except Exception as e:
            sys.stdout.write('\r' + ' ' * 20 + '\r')
            click.echo(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
