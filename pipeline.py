# src/pipeline.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

# ===== MODEL IDS =====
QUEUE_MODEL_ID = "engineermustafahusain/ticket-queue-classifier"
PRIORITY_MODEL_ID = "engineermustafahusain/priority_model"
GEN_MODEL_ID = "google/flan-t5-base"

# ===== LOAD MODELS =====
queue_tokenizer = AutoTokenizer.from_pretrained(QUEUE_MODEL_ID)
queue_model = AutoModelForSequenceClassification.from_pretrained(QUEUE_MODEL_ID)

priority_tokenizer = AutoTokenizer.from_pretrained(PRIORITY_MODEL_ID)
priority_model = AutoModelForSequenceClassification.from_pretrained(PRIORITY_MODEL_ID)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

generator = hf_pipeline(
    "text2text-generation",
    model=GEN_MODEL_ID,
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=generator)

# ===== HELPERS =====
def predict_label(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(dim=1).item()
    return model.config.id2label[pred_id]

def load_vectorstore(queue_name):
    folder = queue_name.lower().replace(" ", "_")
    path = f"src/vectorstore/{folder}"
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ===== MAIN PIPELINE =====
def run_pipeline(text):
    queue = predict_label(text, queue_tokenizer, queue_model)
    priority = predict_label(text, priority_tokenizer, priority_model)

    vectorstore = load_vectorstore(queue)
    docs = vectorstore.similarity_search(text, k=3)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are a support assistant.

Context:
{context}

Question:
{text}

Answer clearly and professionally.
"""

    answer = llm.invoke(prompt)

    return {
        "queue": queue,
        "priority": priority,
        "answer": answer
    }
