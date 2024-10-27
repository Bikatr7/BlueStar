import click
from utils.retrieval import Retriever
from utils.generation import RAGModel

@click.command()
@click.option('--model-path', default='../models/quantized-mistral-7b', help='Path to the quantized model.')
@click.option('--index-path', default='../data/faiss_index.bin', help='Path to FAISS index.')
@click.option('--corpus-path', default='../data/corpus.pkl', help='Path to the corpus pickle file.')
def main(model_path, index_path, corpus_path):
    try:
        retriever = Retriever(index_path, corpus_path)
        rag = RAGModel(model_path, retriever)
    except Exception as e:
        click.echo(f"Error initializing BlueStar: {e}")
        return

    click.echo("BlueStar RAG CLI. Type 'exit' to quit.")
    while True:
        try:
            query = click.prompt('You', type=str)
            if query.lower() in ['exit', 'quit']:
                break
            response, sources = rag.generate_response(query)
            click.echo(f"BlueStar: {response}")
            if sources:
                click.echo("Sources:")
                for i, doc in enumerate(sources, 1):
                    click.echo(f"{i}. {doc[:200]}...")  ## Display first 200 chars of each source
        except Exception as e:
            click.echo(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
