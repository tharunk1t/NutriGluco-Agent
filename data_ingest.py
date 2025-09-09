
---

## data_ingest.py
Creates chunks, embeddings, and a FAISS vector store. Run once after populating `sample_docs/`.

```python
# data_ingest.py
import os
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import pickle
import pandas as pd

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

CHUNK_SIZE = 400  # characters
CHUNK_OVERLAP = 80

def load_text_files(docs_dir):
    docs = []
    for p in sorted(Path(docs_dir).glob("*")):
        if p.is_file() and p.suffix.lower() in [".txt", ".md", ".csv"]:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
                # Heuristic: join rows or use specified column 'text'
                if "text" in df.columns:
                    texts = df["text"].astype(str).tolist()
                else:
                    texts = df.astype(str).agg(" ".join, axis=1).tolist()
                for i, t in enumerate(texts):
                    docs.append({"source": str(p), "id": f"{p.name}-{i}", "text": t})
            else:
                text = p.read_text(encoding="utf8")
                docs.append({"source": str(p), "id": p.name, "text": text})
    return docs

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Creates embeddings numpy array using sentence-transformers
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def main(docs_dir, index_path):
    docs = load_text_files(docs_dir)
    print(f"Loaded {len(docs)} source files")

    # chunk
    records = []
    for d in docs:
        chunks = chunk_text(d["text"])
        for i, c in enumerate(chunks):
            records.append({"source": d["source"], "chunk_id": f"{d['id']}-{i}", "text": c})

    print(f"Created {len(records)} chunks")

    texts = [r["text"] for r in records]
    embeddings = build_embeddings(texts)

    # Build FAISS index (L2 normalized)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cos sim) with normalized embeddings
    # normalize embeddings
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"FAISS index size: {index.ntotal}")

    # Save index and metadata
    faiss.write_index(index, index_path + ".index")
    with open(index_path + "_meta.pkl", "wb") as f:
        pickle.dump(records, f)

    print("Saved index and metadata.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, default="sample_docs")
    parser.add_argument("--index_path", type=str, default="faiss_index")
    args = parser.parse_args()
    main(args.docs_dir, args.index_path)
