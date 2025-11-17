"""
RAG Chatbot for Patient Aftercare Queries
Runs locally with Gemma-3-1B using GPU auto-detection + 4-bit quantization.
"""

import os
import torch
from typing import List, Dict
from dotenv import load_dotenv
from semantic_search import SemanticSearchEngine
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils import is_flash_attn_2_available

load_dotenv()


class PatientAftercareChatbot:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_id: str = "google/gemma-3-1b-it",
    ):
        """Initialize RAG Chatbot with FAISS + local Gemma model."""
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

        # ===== GPU AUTO-DETECTION =====
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(f"\n✅ GPU detected: {gpu_name} (Compute Capability {capability})")

            # GTX 1650 -> Compute 7.5, no Flash Attn 2
            if is_flash_attn_2_available() and capability[0] >= 8:
                attn_implementation = "flash_attention_2"
                print("⚡ Using Flash Attention 2 for faster inference.")
            else:
                attn_implementation = "sdpa"
                print("💡 Flash Attention 2 not supported — using SDPA instead.")
        else:
            print("\n⚠️ No GPU detected! Falling back to CPU (very slow).")
            attn_implementation = "eager"

        # ===== Quantization (4-bit) =====
        print("\n🔧 Loading Gemma model with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # ===== Load model & tokenizer =====
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=False,
            attn_implementation=attn_implementation,
            device_map="auto",
        )

        # ===== System prompt =====
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

    def generate_response(self, query: str, context: str) -> str:
        """Generate a local LLM response using the Gemma model."""
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

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.4,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response

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
        print("PATIENT AFTERCARE CHATBOT — Local Gemma 3-1B (GPU Adaptive)")
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
    chatbot = PatientAftercareChatbot(model_id="google/gemma-3-1b-it")
    chatbot.chat("How should I care for my wound after surgery?")
    chatbot.interactive_mode()


if __name__ == "__main__":
    main()
