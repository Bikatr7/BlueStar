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

## Add threading.Event for controlling the spinner
def spinner_task(stop_event):
    """Animated spinner to show the model is working"""
    spinner = cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    while not stop_event.is_set():
        sys.stdout.write(f"\rThinking {next(spinner)} ")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * 20 + '\r')
    sys.stdout.flush()

@click.command()
@click.option('--model-path',
    default=os.path.join(os.path.dirname(__file__), "..", "models", "quantized-gpt2-large"),
    help='Path to quantized model.')
@click.option('--index-path',
    default=os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index.bin"),
    help='Path to FAISS index.')
@click.option('--corpus-path',
    default=os.path.join(os.path.dirname(__file__), "..", "data", "corpus.pkl"),
    help='Path to the corpus pickle file.')
@click.option('--device',
    default='cpu',
    help='Device to use for model inference. Currently only supports CPU.')
def main(model_path, index_path, corpus_path, device):
    try:
        click.echo("Initializing BlueStar...")
        retriever = Retriever(index_path, corpus_path)
        rag = RAGModel(model_path, retriever, device)
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
                
            ## Create stop event and start spinner
            stop_spinner = threading.Event()
            spinner = threading.Thread(target=spinner_task, args=(stop_spinner,))
            spinner.daemon = True
            spinner.start()
            
            start_time = time.time()
            cpu_start, ram_start = monitor_resources()
            
            ## Generate response
            try:
                response, sources = rag.generate_response(query)
            except RuntimeError as e:
                stop_spinner.set()
                if "out of memory" in str(e):
                    click.echo("Error: Out of memory. Please try a shorter query or free up system resources.")
                    continue
                else:
                    click.echo(f"An error occurred during generation: {str(e)}")
                    continue
            
            ## Get resource usage
            cpu_end, ram_end = monitor_resources()
            end_time = time.time()
            
            ## Stop the spinner
            stop_spinner.set()
            spinner.join()
            
            click.echo(f"BlueStar: {response}")
            if sources:
                click.echo("\nSources:")
                for i, doc in enumerate(sources, 1):
                    click.echo(f"{i}. {doc[:200]}...")
            
            click.echo(f"\nPerformance Metrics:")
            click.echo(f"Response Time: {end_time - start_time:.2f}s")
            click.echo(f"CPU Usage: {cpu_end - cpu_start:.1f}%")
            click.echo(f"RAM Usage: {ram_end - ram_start:.1f}%")
                    
        except Exception as e:
            stop_spinner.set()
            click.echo(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
