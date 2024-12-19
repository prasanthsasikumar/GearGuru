from fastapi import FastAPI
from pydantic import BaseModel
from query_answer import search_in_faiss, generate_answer
from embed_text import chunk_text

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    question = request.question

    # Load the extracted text and chunk it
    with open("extracted_text.txt", "r") as file:
        text = file.read()
    chunks = chunk_text(text)

    # Search for relevant chunks
    relevant_chunks_idx = search_in_faiss(question, index)
    relevant_chunks = [chunks[i] for i in relevant_chunks_idx[0]]

    # Generate an answer
    answer = generate_answer(question, relevant_chunks)
    return {"answer": answer}
