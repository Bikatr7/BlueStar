from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import textwrap
from BlueStar.utils.retrieval import Retriever

class RAGModel:
    def __init__(self, model_path: str, retriever: Retriever, device: str = 'cpu'):
        try:
            print(f"[INFO] [generation.py] Loading model from {model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("[INFO] [generation.py] Loading quantized model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2,
                return_full_text=False,
                max_new_tokens=150,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("[INFO] [generation.py] Model loaded successfully")
            
        except Exception as e:
            raise Exception(f"[ERROR] [generation.py] Failed to load model: {str(e)}")
            
        self.retriever = retriever
        self.device = device
        self.COLUMN_WIDTH = 76
        self.MAX_INPUT_LENGTH = 800

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
            retrieved_docs = self.retriever.retrieve(query, top_k)
            
            context_parts = []
            remaining_length = self.MAX_INPUT_LENGTH - len(self.tokenizer.encode(query)) - 100
            per_doc_length = remaining_length // min(2, len(retrieved_docs))
            
            for doc in retrieved_docs[:2]:
                doc_excerpt = self.truncate_text(doc, per_doc_length)
                context_parts.append(doc_excerpt)
            
            context = "\n".join(context_parts)
            
            prompt = f"""Using the following reference information:
{context}

Question: {query}
Please provide a clear, focused answer that directly addresses the question:"""
            outputs = self.pipeline(
                prompt,
                do_sample=True,
                max_new_tokens=150, 
                num_return_sequences=1,
                temperature=0.7, 
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            if not outputs:
                print("[WARNING] [generation.py] No response generated")
                return "I apologize, but I couldn't generate a response.", []
                
            response = outputs[0]['generated_text'] if isinstance(outputs[0], dict) else outputs[0]
            response = self.clean_text(response)
            response = self.wrap_text(response)
            
            return response, retrieved_docs[:2]
            
        except Exception as e:
            print(f"[ERROR] [generation.py] Error during generation: {str(e)}")
            return f"An error occurred during generation: {str(e)}", []
    
    def is_allowed_topic(self, query: str) -> bool:
        """Check if query is about allowed topics"""
        disallowed = [
            'violence', 'illegal', 'hate speech', 'adult content',
            'weapons', 'drugs', 'terrorism', 'extremism',
            'personal information', 'hacking', 'malware'
        ]

        if any(topic in query.lower() for topic in disallowed):
            return False

        if any(cmd in query.lower() for cmd in ['sudo', 'rm -rf', 'format', 'del']):
            return False
                
        return True

    def refine_query(self, query: str) -> str:
        ## remains to be implemented properly, i'm pretty sure it's fucking up shit#
        """Improve query clarity if needed"""
        if len(query.strip()) < 5:
            return f"Could you please elaborate? Your message '{query}' is too short for me to understand clearly."
                
        common_responses = ['ok', 'okay', 'yes', 'no', 'thanks']
        if query.lower().strip() in common_responses:
            return query

        vague_terms = ['this', 'that', 'it', 'thing']
        if any(term in query.lower().split() for term in vague_terms):
            return f"Could you be more specific about what you mean by '{query}'?"
                
        if any(term in query.lower() for term in ['code', 'program', 'function']):
            return f"Regarding '{query}', what specific aspect or language are you interested in?"
                
        return query
