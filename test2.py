"""
RAG Chatbot for Patient Aftercare Queries
Uses FAISS + Hugging Face Inference API (Mixtral-8x7B-Instruct) for response generation.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from semantic_search import SemanticSearchEngine
from huggingface_hub import InferenceClient
import traceback

# Load environment variables
load_dotenv()


class PatientAftercareChatbot:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        hf_api_key: str = None,
    ):
        """Initialize RAG Chatbot."""
        print("Initializing Semantic Search Engine...")
        self.search_engine = SemanticSearchEngine(model_name=embedding_model)

        # Load or build FAISS index
        if os.path.exists("faiss_index.bin") and os.path.exists("chunks_data.pkl"):
            self.search_engine.load_index()
        else:
            self.search_engine.load_chunks("chunks.json")
            embeddings = self.search_engine.create_embeddings()
            self.search_engine.build_index(embeddings)
            self.search_engine.save_index()

        # Hugging Face client
        self.model_name = model_name
        self.hf_api_key = hf_api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.hf_api_key:
            raise ValueError("Missing Hugging Face API token. Add it to .env or environment variables.")
        print(f"Hugging Face Hosted Model: {self.model_name}")
        print("Connecting to Hugging Face Inference API...")

        self.client = InferenceClient(token=self.hf_api_key, provider="hf-inference")

        # System prompt
        self.system_prompt = """You are a helpful medical assistant specializing in post-operative patient care.

Responsibilities:
1. Provide accurate, evidence-based information from the retrieved documents.
2. Prioritize patient safety and clarity.
3. Highlight emergency red flags clearly.
4. Avoid making diagnoses; provide aftercare guidance only.
5. If information is missing, say so clearly.
6. Cite the source documents whenever possible.
7. Encourage patients to contact their healthcare provider.

Tone: calm, supportive, and easy to understand.
"""

    def retrieve_context(self, query: str, top_k: int = 5, window: int = 2) -> tuple[str, List[Dict]]:
        """Retrieve relevant context using FAISS sentence-level search."""
        results = self.search_engine.search(query, top_k=top_k, window=window)
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Document {i}]\nSource: {r['source']}\nRelevance: {r['similarity_score']:.3f}\nContent: {r['content']}\n"
            )
        formatted_context = "\n---\n".join(context_parts)
        return formatted_context, results

    import traceback

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Hugging Face hosted chat model with Zephyr fallback."""

        prompt = f"""{self.system_prompt}

Retrieved Documents:
{context}

Patient Question: {query}

Please include:
1. A direct, patient-friendly answer
2. Safety and red flag information
3. When to contact a healthcare provider
4. References to source documents
"""

        try:
        # Primary model: Mixtral
            response = self.client.chat.completions.create(
                model=self.model_name,  # mistralai/Mixtral-8x7B-Instruct-v0.1
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.3,
            )

            return response.choices[0].message["content"].strip()

        except Exception as e:
            print("\n⚠️ Mixtral model failed — switching to Zephyr fallback.\n")
            print("Error details:", str(e))
            traceback.print_exc()

        # Fallback model: Zephyr
        try:
            response = self.client.chat.completions.create(
                model="HuggingFaceH4/zephyr-7b-beta",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.3,
            )
            return response.choices[0].message["content"].strip()

        except Exception as e2:
            print("\n🚨 Fallback Zephyr model also failed.")
            print("Error details:", str(e2))
            traceback.print_exc()
            return (
                "I'm sorry, but I’m unable to process your request right now. "
                "Please try again in a few minutes."
            )


    def chat(self, query: str, top_k: int = 5, verbose: bool = True) -> Dict:
        """Process user query."""
        if verbose:
            print(f"\n{'='*80}\nQuery: {query}\n{'='*80}\nRetrieving relevant documents...")

        context, results = self.retrieve_context(query, top_k=top_k)
        if verbose:
            print(f"Retrieved {len(results)} relevant documents.\n")

        response = self.generate_response(query, context)

        if verbose:
            print(f"{'='*80}\nRESPONSE:\n{'='*80}\n{response}\n")

        return {"query": query, "response": response, "retrieved_documents": results}

    def interactive_mode(self):
        """CLI loop."""
        print("\n" + "="*80)
        print("PATIENT AFTERCARE CHATBOT — Mixtral-8x7B (Hugging Face)")
        print("="*80)
        print("Ask any post-operative care question. Type 'exit' to quit.\n")

        while True:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye! Always follow your doctor's instructions.")
                break
            if not query:
                continue
            result = self.chat(query, verbose=False)
            print(f"\nAssistant: {result['response']}\n")


def main():
    chatbot = PatientAftercareChatbot(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        hf_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    chatbot.chat("How should I care for my wound after surgery?")
    chatbot.interactive_mode()


if __name__ == "__main__":
    main()
