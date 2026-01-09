import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


# =========================
# CONFIG
# =========================
QUEUE_MODEL_ID = "engineermustafahusain/ticket-queue-classifier"
PRIORITY_MODEL_ID = "engineermustafahusain/priority_model"

VECTORSTORE_BASE_PATH = "src/vectorstore"

DEVICE = 0 if torch.cuda.is_available() else -1


# =========================
# LOAD MODELS (CACHED)
# =========================
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(QUEUE_MODEL_ID)

    queue_model = AutoModelForSequenceClassification.from_pretrained(
        QUEUE_MODEL_ID
    )

    priority_model = AutoModelForSequenceClassification.from_pretrained(
        PRIORITY_MODEL_ID
    )

    return tokenizer, queue_model, priority_model


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        device=DEVICE,
        max_new_tokens=200
    )
    return HuggingFacePipeline(pipeline=generator)


# =========================
# QUEUE → VECTORSTORE
# =========================
def load_faiss_for_queue(queue_name, embeddings):
    folder_name = queue_name.lower().replace(" ", "_")
    path = f"{VECTORSTORE_BASE_PATH}/{folder_name}"

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================
# PREDICTION LOGIC
# =========================
def predict_queue(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(dim=-1).item()
    return model.config.id2label[pred_id]


def predict_priority(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(dim=-1).item()
    return model.config.id2label[pred_id]


def rag_answer(question, queue_name, embeddings, llm):
    db = load_faiss_for_queue(queue_name, embeddings)
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a customer support assistant.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""

    return llm.invoke(prompt)


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="AI Ticket Classifier",
    layout="centered"
)

st.title("🎫 AI Ticket Classifier + RAG")
st.write("Predict **Queue**, **Priority**, and get a **knowledge-based answer**")

email_text = st.text_area(
    "Paste customer email",
    height=200
)

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter email text")
    else:
        with st.spinner("Analyzing ticket..."):

            tokenizer, queue_model, priority_model = load_models()
            embeddings = load_embeddings()
            llm = load_llm()

            queue = predict_queue(email_text, tokenizer, queue_model)
            priority = predict_priority(email_text, tokenizer, priority_model)
            answer = rag_answer(email_text, queue, embeddings, llm)

        st.success("Prediction Complete")

        st.markdown(f"### 📂 Queue: `{queue}`")
        st.markdown(f"### 🚦 Priority: `{priority}`")

        st.markdown("### 🤖 Suggested Answer")
        st.write(answer)
