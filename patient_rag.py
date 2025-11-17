import os
import json
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------
# Optional: disable symlink warning
# ------------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ------------------------------
# STEP 1: Load JSON Chunks
# ------------------------------
with open("chunk/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["content"] for item in data if item.get("content")]
print(f"✅ Loaded {len(texts)} text chunks.")

# ------------------------------
# STEP 2: Create Embeddings
# ------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ------------------------------
# STEP 3: Create / Save FAISS
# ------------------------------
vectorstore = FAISS.from_texts(texts, embedding=embeddings)
vectorstore.save_local("patient_aftercare_faiss")
print("✅ FAISS vector store created and saved.")

# ------------------------------
# STEP 4: Retriever
# ------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ------------------------------
# STEP 5: Load Local Open-Source LLM
# ------------------------------
model_name = "declare-lab/flan-alpaca-base"  # CPU-friendly, instruction-tuned
print("⏳ Loading LLM model, please wait...")

# Explicitly set tokenizer with clean_up_tokenization_spaces=True
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

llm_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    min_length=50,
    temperature=0.3
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)
print("✅ Model loaded successfully.")

# ------------------------------
# STEP 6: Custom Prompt
# ------------------------------
prompt_template = """
You are a friendly and knowledgeable patient after-care assistant.
Use the provided context (from verified hospital documents) to answer the user's question clearly.

Be:
- Empathetic and professional
- Detailed but easy to understand
- Cautious — if unsure, recommend the user consult their doctor

If the context does not contain enough information, say:
"I'm sorry, I recommend speaking to your healthcare provider for accurate advice."

Context:
{context}

Question:
{question}

Answer:
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# ------------------------------
# STEP 7: Build RAG Chain
# ------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# ------------------------------
# STEP 8: Chat Loop
# ------------------------------
print("\n🩺 Patient After-Care Assistant Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye! Stay healthy.")
        break

    # Use .invoke() instead of deprecated .run()
    response = qa_chain.invoke({"query": query})
    print("Bot:", response)
