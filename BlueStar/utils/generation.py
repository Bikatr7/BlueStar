from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.retrieval import Retriever

class RAGModel:
    def __init__(self, model_path: str, retriever: Retriever, device: str = 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
        self.retriever = retriever
        self.device = device
        self.disallowed_topics = ['violence', 'illegal', 'hate', 'privacy']
        self.max_context_length = 2048
        self.system_prompt = """You are a helpful AI assistant. Always provide accurate, 
        factual information based on the given context. If you're unsure or the context 
        doesn't contain relevant information, say so."""

    def is_allowed(self, query: str):
        query_lower = query.lower()
        for topic in self.disallowed_topics:
            if topic in query_lower:
                return False
        return True

    def generate_response(self, query: str, top_k: int = 5, max_length: int = 150):
        if not self.is_allowed(query):
            return "I apologize, but I cannot assist with that topic due to ethical constraints.", []

        # Retrieve and format relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k)
        formatted_docs = []
        for i, doc in enumerate(retrieved_docs, 1):
            formatted_docs.append(f"[Document {i}]: {doc}")
        
        context = "\n\n".join(formatted_docs)
        
        ## Construct prompt with system instruction and context
        prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', 
                                         truncation=True, 
                                         max_length=self.max_context_length).to(self.device)
            
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            ## Format response with source citations
            final_response = f"{response}\n\nSources:"
            for i, doc in enumerate(retrieved_docs, 1):
                final_response += f"\n{i}. {doc[:100]}..."
                
            return final_response, retrieved_docs
            
        except Exception as e:
            return f"An error occurred while generating the response: {str(e)}", []
