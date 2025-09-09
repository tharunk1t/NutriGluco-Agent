# app.py
import os
import threading
import time
import requests
import gradio as gr
from fastapi import FastAPI
import uvicorn

from agent import NutritionRAG

# create rag instance (loads models/index)
rag = NutritionRAG(index_path="faiss_index")

# FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def post_query(payload: dict):
    q = payload.get("query", "")
    if not q:
        return {"answer": "No query provided."}
    ans = rag.answer(q)
    return {"answer": ans}

# Gradio UI using local backend call (so the UI is decoupled)
def gradio_query(user_input, chat_history):
    if chat_history is None:
        chat_history = []
    try:
        r = requests.post("http://127.0.0.1:8000/query", json={"query": user_input}, timeout=20)
        ans = r.json().get("answer", "No answer.")
    except Exception as e:
        ans = f"Backend error: {e}"
    chat_history.append((user_input, ans))
    return chat_history, chat_history

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # start FastAPI in a background thread
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(1.5)
    # Launch Gradio (blocks)
    with gr.Blocks() as demo:
        gr.Markdown("# Diabetes Nutrition RAG — Assistant")
        chatbot = gr.Chatbot()
        txt = gr.Textbox(placeholder="e.g., 'Is banana OK if my blood glucose is 180 mg/dL?'")
        state = gr.State([])
        txt.submit(gradio_query, [txt, state], [chatbot, state])
        gr.Button("Clear").click(lambda : ([], []), None, [chatbot, state])
    demo.launch(share=True)
