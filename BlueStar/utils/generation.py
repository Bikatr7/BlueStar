import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import textwrap
from BlueStar.utils.retrieval import Retriever

class RAGModel:
    def __init__(self, model_path: str, retriever: Retriever, device: str = 'auto'):
        try:
            ## Load from Hugging Face
            model_id = "gpt2-large"  ## 774M parameters
            print(f"Loading model from {model_id}...")
            
            ## Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            ## Load model with CPU optimizations
            print("Loading model with CPU optimizations...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            ## Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=False,  ## Only return the generated text
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("Model loaded successfully")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
            
        self.retriever = retriever
        self.device = device
        self.COLUMN_WIDTH = 76
        self.MAX_INPUT_LENGTH = 800  ## Leave room for generation within 1024 token limit

    def clean_text(self, text: str) -> str:
        """Remove newlines and special characters"""
        return text.replace('\n', ' ').replace('\r', '').strip()

    def wrap_text(self, text: str) -> str:
        """Wrap text to specified column width"""
        return textwrap.fill(text, width=self.COLUMN_WIDTH)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate_response(self, query: str, top_k: int = 3) -> tuple[str, list]:
        try:
            ## Get relevant documents
            retrieved_docs = self.retriever.retrieve(query, top_k)
            
            ## Take shorter excerpts from each document
            context_parts = []
            remaining_length = self.MAX_INPUT_LENGTH - len(self.tokenizer.encode(query)) - 100  # Reserve tokens for query and format
            per_doc_length = remaining_length // min(2, len(retrieved_docs))
            
            for doc in retrieved_docs[:2]:  ## Use top 2 documents
                doc_excerpt = self.truncate_text(doc, per_doc_length)
                context_parts.append(doc_excerpt)
            
            context = "\n".join(context_parts)
            
            ## Create prompt formatted for GPT-2
            prompt = f"""Based on this information:
{context}

Question: {query}
Answer:"""
            
            ## Final truncation check
            prompt = self.truncate_text(prompt, self.MAX_INPUT_LENGTH)
            
            ## Generate using pipeline
            outputs = self.pipeline(
                prompt,
                do_sample=True,
                max_new_tokens=100,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            if not outputs:
                return "I apologize, but I couldn't generate a response.", []
                
            ## Clean and format response
            response = outputs[0]['generated_text'] if isinstance(outputs[0], dict) else outputs[0]
            response = self.clean_text(response)
            response = self.wrap_text(response)
            
            return response, retrieved_docs[:2]  # Return top 2 relevant documents
            
        except Exception as e:
            return f"An error occurred during generation: {str(e)}", []

    def is_allowed_topic(self, query: str) -> bool:
        """Check if query is about allowed topics"""
        disallowed = ['violence', 'illegal', 'hate speech', 'adult content']
        return not any(topic in query.lower() for topic in disallowed)

    def refine_query(self, query: str) -> str:
        """Improve query clarity if needed"""
        if len(query) < 10:
            return f"Please elaborate on: {query}"
        return query
