import faiss
import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# Initialize LLaMA model (1B or 3B)
model_name = "huggingface/llama-3b"  # Choose 'llama-1b' or 'llama-3b'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

def chunk_text(text):
    """
    Splits text into chunks for processing.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_embedding(text):
    """
    Creates an embedding for the input text using LLaMA.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().cpu().numpy()

def create_faiss_index(embeddings):
    """
    Creates and stores embeddings in a FAISS index.
    """
    dim = embeddings[0].shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dim)
    embeddings_np = np.vstack(embeddings)
    index.add(embeddings_np)
    return index

def save_faiss_index(index, file_name="motorcycle_service_manual.index"):
    """
    Saves the FAISS index to a file.
    """
    faiss.write_index(index, file_name)

if __name__ == "__main__":
    # Load extracted text
    with open("extracted_text.txt", "r") as file:
        text = file.read()

    # Chunk the text
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = [create_embedding(chunk) for chunk in chunks]

    # Create FAISS index
    faiss_index = create_faiss_index(embeddings)

    # Save the FAISS index
    save_faiss_index(faiss_index)
    print("FAISS index creation and saving complete!")
