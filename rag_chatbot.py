"""
RAG Chatbot for Patient Aftercare Queries
Uses semantic search with FAISS and LLM for response generation
"""

import os
from typing import List, Dict
from semantic_search import SemanticSearchEngine
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PatientAftercareChatbot:
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_provider: str = "anthropic",  # "openai" or "anthropic"
                 model_name: str = None):
        """
        Initialize RAG Chatbot
        
        Args:
            embedding_model: Model for creating embeddings
            llm_provider: "openai" or "anthropic"
            model_name: Specific model name (defaults to best available)
        """
        # Initialize semantic search engine
        print("Initializing Semantic Search Engine...")
        self.search_engine = SemanticSearchEngine(model_name=embedding_model)
        
        # Load or build index
        if os.path.exists('faiss_index.bin') and os.path.exists('chunks_data.pkl'):
            self.search_engine.load_index()
        else:
            self.search_engine.load_chunks('chunks.json')
            embeddings = self.search_engine.create_embeddings()
            self.search_engine.build_index(embeddings)
            self.search_engine.save_index()
        
        # Initialize LLM
        self.llm_provider = llm_provider
        
        if llm_provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = model_name or "gpt-4-turbo-preview"
        elif llm_provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = model_name or "claude-3-5-sonnet-20241022"
        else:
            raise ValueError("llm_provider must be 'openai' or 'anthropic'")
        
        print(f"LLM Provider: {llm_provider}")
        print(f"Model: {self.model_name}")
        
        # System prompt for the chatbot
        self.system_prompt = """You are a helpful medical assistant specializing in post-operative patient care and aftercare instructions. 

Your role is to:
1. Provide accurate, evidence-based information from the retrieved medical documents
2. Always prioritize patient safety
3. Clearly indicate when medical attention is urgently needed
4. Use clear, compassionate language appropriate for patients
5. Never provide diagnoses - only aftercare guidance based on the documents
6. If information is not in the retrieved documents, clearly state this

Important guidelines:
- Always cite the source documents when providing information
- Emphasize red flags and emergency situations
- Encourage patients to contact their healthcare provider with concerns
- Be specific about medication instructions, wound care, and activity restrictions
- Provide practical, actionable advice

Remember: You are providing educational information, not medical advice. Patients should always follow their healthcare provider's specific instructions."""

    def retrieve_context(self, query: str, top_k: int = 5) -> tuple[str, List[Dict]]:
        """
        Retrieve relevant context chunks for the query
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (formatted_context, raw_results)
        """
        results = self.search_engine.search(query, top_k=top_k)
        
        # Format context for LLM
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {result['source']}\n"
                f"Relevance Score: {result['similarity_score']:.3f}\n"
                f"Content: {result['content']}\n"
            )
        
        formatted_context = "\n---\n".join(context_parts)
        return formatted_context, results
    
    def generate_response_openai(self, query: str, context: str) -> str:
        """Generate response using OpenAI"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Based on the following medical documents about post-operative care, please answer the patient's question.

Retrieved Documents:
{context}

Patient Question: {query}

Please provide a comprehensive, patient-friendly answer based on the retrieved documents. Include:
1. Direct answer to their question
2. Relevant safety information and red flags
3. When to contact healthcare provider
4. References to source documents"""}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def generate_response_anthropic(self, query: str, context: str) -> str:
        """Generate response using Anthropic Claude"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=2000,
            temperature=0.3,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"""Based on the following medical documents about post-operative care, please answer the patient's question.

Retrieved Documents:
{context}

Patient Question: {query}

Please provide a comprehensive, patient-friendly answer based on the retrieved documents. Include:
1. Direct answer to their question
2. Relevant safety information and red flags
3. When to contact healthcare provider
4. References to source documents"""
                }
            ]
        )
        
        return message.content[0].text
    
    def chat(self, query: str, top_k: int = 5, verbose: bool = True) -> Dict:
        """
        Process a query and generate response
        
        Args:
            query: User query
            top_k: Number of context chunks to retrieve
            verbose: Print detailed information
            
        Returns:
            Dictionary with response and metadata
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}\n")
            print("Retrieving relevant documents...")
        
        # Retrieve context
        context, results = self.retrieve_context(query, top_k=top_k)
        
        if verbose:
            print(f"Retrieved {len(results)} relevant documents\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['source']} (Score: {result['similarity_score']:.3f})")
            print(f"\n{'='*80}")
            print("Generating response...\n")
        
        # Generate response
        if self.llm_provider == "openai":
            response = self.generate_response_openai(query, context)
        else:
            response = self.generate_response_anthropic(query, context)
        
        if verbose:
            print(f"{'='*80}")
            print("RESPONSE:")
            print(f"{'='*80}\n")
            print(response)
            print(f"\n{'='*80}\n")
        
        return {
            'query': query,
            'response': response,
            'retrieved_documents': results,
            'num_sources': len(results)
        }
    
    def interactive_mode(self):
        """Run chatbot in interactive mode"""
        print("\n" + "="*80)
        print("PATIENT AFTERCARE CHATBOT - Interactive Mode")
        print("="*80)
        print("\nWelcome! I'm here to help you with post-operative care questions.")
        print("I can provide information about:")
        print("  • Post-surgery instructions and recovery")
        print("  • Pain management and medications")
        print("  • Wound care and infection prevention")
        print("  • Warning signs and when to seek help")
        print("  • Activity restrictions and return to normal life")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("="*80 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                query = input("\nYour Question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Patient Aftercare Chatbot.")
                    print("Remember: Always follow your healthcare provider's specific instructions!")
                    break
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Process query
                result = self.chat(query, top_k=5, verbose=False)
                
                print(f"\n{'='*80}")
                print("RESPONSE:")
                print(f"{'='*80}\n")
                print(result['response'])
                
                # Show sources
                print(f"\n{'='*80}")
                print("SOURCES:")
                print(f"{'='*80}")
                for i, doc in enumerate(result['retrieved_documents'], 1):
                    print(f"{i}. {doc['source']} (Relevance: {doc['similarity_score']:.3f})")
                print(f"{'='*80}\n")
                
                conversation_history.append(result)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again or type 'quit' to exit.")


def main():
    """Main function to run the chatbot"""
    
    # Example queries for demonstration
    example_queries = [
        "What should I do if I have pain after knee replacement surgery?",
        "When can I return to work after having a C-section?",
        "What are the red flags after appendectomy that I should watch for?",
        "How should I care for my surgical wound?",
        "Is it normal to have a fever after surgery?"
    ]
    
    print("\n" + "="*80)
    print("INITIALIZING PATIENT AFTERCARE RAG CHATBOT")
    print("="*80)
    
    # Initialize chatbot (choose your LLM provider)
    # For OpenAI:
    # chatbot = PatientAftercareChatbot(llm_provider="openai")
    
    # For Anthropic Claude (recommended for medical use cases):
    chatbot = PatientAftercareChatbot(llm_provider="anthropic")
    
    print("\n" + "="*80)
    print("RUNNING EXAMPLE QUERIES")
    print("="*80)
    
    # Run example queries
    for query in example_queries[:2]:  # Run first 2 examples
        chatbot.chat(query, top_k=5, verbose=True)
    
    # Start interactive mode
    chatbot.interactive_mode()


if __name__ == "__main__":
    main()
