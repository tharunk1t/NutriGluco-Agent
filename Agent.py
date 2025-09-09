# agent.py
import os
import pickle
import faiss
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Config
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "google/flan-t5-base")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Load tokenizer & model helper (with fallback)
def load_text_generator(model_id=HF_MODEL_ID):
    try:
        # pass token if present
        token_arg = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **token_arg)
        model = AutoModelForCausalLM.from_pretrained(model_id, **token_arg)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.2, trust_remote_code=True)
        print("Loaded model:", model_id)
        return pipe
    except Exception as e:
        print("Failed to load requested model:", e)
        print("Falling back to:", FALLBACK_MODEL)
        pipe = pipeline("text2text-generation", model=FALLBACK_MODEL)
        return pipe

# Retriever: loads faiss + meta and does k-nearest
def load_retriever(index_path="faiss_index"):
    idx = faiss.read_index(index_path + ".index")
    with open(index_path + "_meta.pkl", "rb") as f:
        records = pickle.load(f)
    # we will use sentence-transformers to embed queries (same model as ingestion)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL)
    return idx, records, embed_model

# simple helper to get top k docs
def retrieve(query: str, idx, records, embed_model, top_k=4):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = idx.search(q_emb, top_k)
    hits = []
    for i in I[0]:
        if i < len(records):
            hits.append(records[i])
    return hits

# Build prompt using retrieved context
def build_prompt(query: str, contexts: List[dict]) -> str:
    context_text = "\n\n".join([f"Source: {c['source']}\n{textwrap_shorten(c['text'], 400)}" for c in contexts])
    prompt = (
        "You are a clinical nutrition assistant specialized in diabetes management. "
        "Answer the user's question concisely and use the provided sources to ground your reply. "
        "If the question requires clinical judgement, recommend consulting a healthcare professional.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION: {query}\n\nANSWER:"
    )
    return prompt

def textwrap_shorten(text, max_len=500):
    # basic shortening preserving sentences
    if len(text) <= max_len:
        return text
    # try to cut at sentence boundaries
    last = text[:max_len].rfind(".")
    if last > 100:
        return text[:last+1]
    return text[:max_len] + "..."

# main interface
class NutritionRAG:
    def __init__(self, index_path="faiss_index"):
        self.generator = load_text_generator()
        self.idx, self.records, self.embed_model = load_retriever(index_path)

    def answer(self, query: str, top_k: int = 4) -> str:
        contexts = retrieve(query, self.idx, self.records, self.embed_model, top_k)
        prompt = build_prompt(query, contexts)
        # call LLM
        out = self.generator(prompt, max_new_tokens=256)
        # extract text for both pipeline types
        text = out[0].get("generated_text") or out[0].get("text") or out[0].get("summary_text") or str(out[0])
        # include citations (source filenames)
        sources = ", ".join({c["source"] for c in contexts})
        final = f"{text.strip()}\n\nSources: {sources}"
        return final
