import numpy as np
import requests
import faiss
from typing import List, Optional
from dataclasses import dataclass

OLLAMA_BASE_URL = "http://localhost:11434/api"

@dataclass
class SearchResult:
    text: str
    score: float

class VectorSearch:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.index = None
        self.chunks = []

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API."""
        response = requests.post(
            f"{OLLAMA_BASE_URL}/embeddings",
            json={"model": self.model_name, "prompt": text}
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")
            
        embedding = np.array(response.json()["embedding"], dtype=np.float32)
        return embedding.reshape(1, -1)

    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks

    def create_index(self, text: str) -> None:
        """Create FAISS index from text."""
        # Create chunks
        self.chunks = self.create_chunks(text)
        
        # Get embeddings for all chunks
        embeddings = []
        for chunk in self.chunks:
            try:
                embedding = self.get_embedding(chunk)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error getting embedding: {e}")
                continue

        # Stack embeddings and create index
        embeddings_array = np.vstack(embeddings)
        dimension = embeddings_array.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

    def search(self, query: str, k: int = 3) -> List[SearchResult]:
        """Search for similar chunks."""
        if not self.index:
            raise Exception("Index not created. Call create_index first.")

        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid index
                results.append(SearchResult(
                    text=self.chunks[idx],
                    score=float(distance)
                ))
                
        return results

    def ask(self, query: str) -> str:
        """Get answer from Ollama using relevant context."""
        results = self.search(query)
        if not results:
            return "No relevant information found."

        context = "\n\n".join(r.text for r in results)
        prompt = f"Based on this context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

        response = requests.post(
            f"{OLLAMA_BASE_URL}/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")
            
        return response.json()["message"]["content"]

# Example usage
if __name__ == "__main__":
    # Initialize
    search = VectorSearch(model_name="llama3.2")
    
    # Load text and create index
    with open("extracted_text.txt", "r") as f:
        text = f.read()
    search.create_index(text)
    
    # Example search
    query = "What parts are considered evaporative-related components"
    results = search.search(query)
    for result in results:
        print(f"Score: {result.score:.2f}")
        print(result.text)
        print("---")
    
    # Example Q&A
    answer = search.ask(query)
    print("\nAnswer:", answer)