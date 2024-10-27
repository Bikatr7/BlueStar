import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import textwrap
from BlueStar.utils.retrieval import Retriever

class RAGModel:
    def __init__(self, model_path: str, retriever: Retriever, device: str = 'cpu'):
        try:
            ## Load from Hugging Face
            model_id = "mistralai/Mistral-7B-Instruct-v0.1"
            print(f"Loading model from {model_id}...")
            
            ## Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            ## Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto"
            )
            
            ## Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                temperature=0.85,
                repetition_penalty=1.1,
                return_full_text=True,
                max_new_tokens=256,
                device_map="auto"
            )
            
            print("Model loaded successfully")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
            
        self.retriever = retriever
        self.device = device
        self.COLUMN_WIDTH = 76

    def clean_text(self, text: str) -> str:
        """Remove newlines and special characters"""
        return text.replace('\n', ' ').replace('\r', '').strip()

    def wrap_text(self, text: str) -> str:
        """Wrap text to specified column width"""
        return textwrap.fill(text, width=self.COLUMN_WIDTH)

    def generate_response(self, query: str, top_k: int = 3) -> tuple[str, list]:
        try:
            ## Get relevant documents
            retrieved_docs = self.retriever.retrieve(query, top_k)
            context = "\n".join(retrieved_docs)
            
            ## Create prompt
            prompt = f"""You are an AI assistant. Use the following context to answer the question.

Context: {context}

Question: {query}
Answer:"""
            
            ## Generate using pipeline
            response = self.pipeline(prompt)[0]['generated_text']
            
            ## Clean and format response
            response = response[len(prompt):]
            response = self.clean_text(response)
            response = self.wrap_text(response)
            
            return response, retrieved_docs
            
        except Exception as e:
            return f"An error occurred: {str(e)}", []

    def is_allowed_topic(self, query: str) -> bool:
        """Check if query is about allowed topics"""
        disallowed = ['violence', 'illegal', 'hate speech', 'adult content']
        return not any(topic in query.lower() for topic in disallowed)

    def refine_query(self, query: str) -> str:
        """Improve query clarity if needed"""
        if len(query) < 10:
            return f"Please elaborate on: {query}"
        return query
