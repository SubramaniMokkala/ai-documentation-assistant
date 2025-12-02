# Embeddings Module
# Converts text to vectors and performs semantic search using FAISS

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple


class EmbeddingStore:
    """
    Manages document embeddings and semantic search
    
    What are embeddings?
    - Vectors (lists of numbers) that represent text meaning
    - Similar text = similar vectors
    - Enables semantic search (meaning-based, not just keyword matching)
    
    Example:
    "cat" and "kitten" have similar embeddings even though different words
    "bank" (river) and "bank" (money) have different embeddings (different meanings)
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding store
        
        Args:
            model_name: Pre-trained model from sentence-transformers
            
        Why this model?
        - Fast: ~50ms per sentence
        - Small: ~80MB
        - Good quality: 384-dimensional vectors
        - Free and runs locally
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # Output dimension of the model
        
        # FAISS index (vector database)
        self.index = None
        
        # Store metadata for each chunk
        self.chunks = []
        self.documents = []
        
        print(f"✅ Model loaded! Embedding dimension: {self.dimension}")
    
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Convert text chunks to embeddings
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        
        Returns:
            numpy array of embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Created {len(embeddings)} embeddings")
        return embeddings
    
    
    def build_index(self, documents: List[Dict]):
        """
        Build FAISS index from documents
        
        Args:
            documents: List of processed documents with chunks
        """
        print("\nBuilding vector index...")
        
        # Collect all chunks from all documents
        all_chunks = []
        for doc in documents:
            for chunk in doc['chunks']:
                # Add document info to chunk
                chunk_with_doc = chunk.copy()
                chunk_with_doc['source_document'] = doc['filename']
                chunk_with_doc['file_type'] = doc['file_type']
                all_chunks.append(chunk_with_doc)
        
        if len(all_chunks) == 0:
            print("No chunks to index!")
            return
        
        # Create embeddings
        embeddings = self.create_embeddings(all_chunks)
        
        # Create FAISS index
        # IndexFlatL2 = brute force search using L2 (Euclidean) distance
        # For < 10k vectors, this is fast enough
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.chunks = all_chunks
        self.documents = documents
        
        print(f"Index built with {len(all_chunks)} chunks from {len(documents)} documents")
    
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search for relevant chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built yet! Call build_index() first.")
        
        # Convert query to embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        # D = distances, I = indices
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.chunks))
        )
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx].copy()
                
                # Convert distance to similarity score (0-1)
                # Lower distance = higher similarity
                similarity = 1 / (1 + distance)
                
                chunk['similarity_score'] = float(similarity)
                chunk['rank'] = i + 1
                
                results.append(chunk)
        
        return results
    
    
    def save(self, path: str = 'data/vectorstore'):
        """
        Save index and metadata to disk
        
        Args:
            path: Directory to save files
        """
        if self.index is None:
            print("No index to save!")
            return
        
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(path, 'faiss.index')
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(path, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'documents': self.documents
            }, f)
        
        print(f"Index saved to {path}")
    
    
    def load(self, path: str = 'data/vectorstore'):
        """
        Load index and metadata from disk
        
        Args:
            path: Directory containing saved files
        """
        index_path = os.path.join(path, 'faiss.index')
        metadata_path = os.path.join(path, 'metadata.pkl')
        
        if not os.path.exists(index_path):
            print(f"No saved index found at {path}")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.documents = data['documents']
        
        print(f"Index loaded from {path}")
        print(f"   {len(self.chunks)} chunks from {len(self.documents)} documents")
        return True


# Test the module
if __name__ == "__main__":
    print("Testing Embeddings Module...")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        {
            'filename': 'test_doc.txt',
            'file_type': 'txt',
            'total_chars': 500,
            'num_chunks': 3,
            'chunks': [
                {'id': 0, 'text': 'Python is a programming language used for data science.'},
                {'id': 1, 'text': 'Machine learning models can predict future outcomes.'},
                {'id': 2, 'text': 'Deep learning uses neural networks for complex tasks.'}
            ]
        }
    ]
    
    # Initialize store
    store = EmbeddingStore()
    
    # Build index
    store.build_index(sample_docs)
    
    # Test search
    print("\nTesting search...")
    query = "What is Python used for?"
    results = store.search(query, top_k=2)
    
    print(f"\nQuery: '{query}'")
    print(f"Top {len(results)} results:\n")
    
    for result in results:
        print(f"Rank {result['rank']}: (Similarity: {result['similarity_score']:.3f})")
        print(f"   Text: {result['text']}")
        print()
    
    print("=" * 60)
    print("Embeddings module working correctly!")
    print("\nKey concepts demonstrated:")
    print("  - Text → Vector conversion")
    print("  - Semantic similarity search")
    print("  - FAISS vector database")