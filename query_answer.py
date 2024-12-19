import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM
import faiss
from embed_text import create_embedding, chunk_text

# Load FAISS index
index = faiss.read_index("motorcycle_service_manual.index")

# Initialize LLaMA model
model_name = "huggingface/llama-3b"  # Choose 'llama-1b' or 'llama-3b'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

def search_in_faiss(query, index, k=5):
    """
    Searches the FAISS index for the most relevant chunks.
    """
    query_embedding = create_embedding(query)
    D, I = index.search(query_embedding, k)
    return I  # Indices of the top k relevant chunks

def generate_answer(question, relevant_chunks):
    """
    Uses LLaMA to generate an answer based on the relevant chunks.
    """
    prompt = f"Answer the following question based on the text: {question}\n\n" + "\n\n".join(relevant_chunks)
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=500)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    # Example query
    query = "How do I change the oil in a motorcycle?"

    # Load extracted text and chunk it
    with open("extracted_text.txt", "r") as file:
        text = file.read()
    chunks = chunk_text(text)

    # Search for relevant chunks
    relevant_chunks_idx = search_in_faiss(query, index)
    relevant_chunks = [chunks[i] for i in relevant_chunks_idx[0]]

    # Generate an answer
    answer = generate_answer(query, relevant_chunks)
    print("Answer:", answer)
