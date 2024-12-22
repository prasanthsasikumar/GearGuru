import faiss
import numpy as np
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ollama API Configuration
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"

def chunk_text(text):
    """
    Splits text into chunks for processing.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_embedding(text):
    """
    Creates an embedding for the input text using the Ollama API.
    """
    payload = {
        "model": "llama3.2",  # Specify the model being used
        "input": text
    }

    try:
        # Make the POST request to Ollama's embedding API
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()

        # Extract the embeddings from the response
        result = response.json()
        return np.array(result.get("embedding", []), dtype=np.float32)
    except requests.RequestException as e:
        raise RuntimeError(f"Error querying the Ollama API: {e}")

def create_faiss_index(embeddings):
    """
    Creates and stores embeddings in a FAISS index.
    """
    dim = embeddings[0].shape[0]  # Embedding dimension
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

    # Generate embeddings using Ollama
    embeddings = [create_embedding(chunk) for chunk in chunks]

    # Create FAISS index
    faiss_index = create_faiss_index(embeddings)

    # Save the FAISS index
    save_faiss_index(faiss_index)
    print("FAISS index creation and saving complete!")