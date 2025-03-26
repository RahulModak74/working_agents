#!/usr/bin/env python3

import os
import json
import numpy as np
import faiss
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer


class VectorDB:
    """Vector database for semantic search using FAISS and sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = None,
                 storage_dir: str = "./vector_storage"):
        """Initialize the vector database with the specified model"""
        self.model_name = model_name
        self.storage_dir = storage_dir
        self.encoder = None
        self.index = None
        self.dimension = dimension
        self.metadata = {}  # Maps ID to metadata
        self.id_map = {}    # Maps FAISS internal ID to our external ID
        self.rev_id_map = {}  # Maps our external ID to FAISS internal ID
        self.next_id = 0
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
    def _load_encoder(self):
        """Load the sentence transformer model"""
        if self.encoder is None:
            try:
                self.encoder = SentenceTransformer(self.model_name)
                if self.dimension is None:
                    # Get dimension from the model
                    self.dimension = self.encoder.get_sentence_embedding_dimension()
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    def _create_index(self):
        """Create a new FAISS index"""
        if self.dimension is None:
            self._load_encoder()
        
        # Create a flat index for exact search
        self.index = faiss.IndexFlatL2(self.dimension)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get the embedding vector for a text string"""
        self._load_encoder()
        return self.encoder.encode([text])[0]
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embedding vectors for multiple text strings"""
        self._load_encoder()
        return self.encoder.encode(texts)
    
    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for a text entry"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None,
                  ids: Optional[List[str]] = None) -> List[str]:
        """Add multiple texts to the vector database"""
        if self.index is None:
            self._create_index()
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        if ids is None:
            ids = [self._generate_id(text) for text in texts]
        
        # Get embeddings for all texts
        embeddings = self._get_embeddings(texts)
        
        # Add to FAISS index
        for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
            # Map our document ID to a FAISS internal ID
            internal_id = self.next_id
            self.id_map[internal_id] = doc_id
            self.rev_id_map[doc_id] = internal_id
            self.next_id += 1
            
            # Store metadata
            self.metadata[doc_id] = {
                "text": text,
                "metadata": metadata
            }
            
            # Add vector to index
            self.index.add(np.array([embeddings[i]]))
        
        return ids
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, 
                 doc_id: Optional[str] = None) -> str:
        """Add a single text to the vector database"""
        ids = self.add_texts([text], [metadata] if metadata else None, [doc_id] if doc_id else None)
        return ids[0]
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar texts in the vector database"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search the index
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.id_map:  # -1 means no result
                doc_id = self.id_map[idx]
                if doc_id in self.metadata:
                    results.append({
                        "id": doc_id,
                        "text": self.metadata[doc_id]["text"],
                        "metadata": self.metadata[doc_id]["metadata"],
                        "score": float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                    })
        
        return results
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document from the index"""
        # FAISS doesn't support direct deletion from flat indices
        # We'll rebuild the index without the deleted document
        if doc_id not in self.rev_id_map:
            return False
        
        # Remove from metadata
        if doc_id in self.metadata:
            del self.metadata[doc_id]
        
        # Get internal ID
        internal_id = self.rev_id_map[doc_id]
        
        # Remove from ID maps
        del self.rev_id_map[doc_id]
        del self.id_map[internal_id]
        
        # Rebuild index if needed
        # Note: For a simple implementation, we don't immediately rebuild
        # the FAISS index. Instead, we mark it as deleted in our metadata.
        # A separate method can be called to rebuild the index when needed.
        
        return True
    
    def rebuild_index(self):
        """Rebuild the FAISS index after deletions"""
        if not self.metadata:
            self._create_index()
            return
        
        # Collect all existing documents
        texts = []
        ids = []
        metadatas = []
        
        for doc_id, data in self.metadata.items():
            texts.append(data["text"])
            metadatas.append(data["metadata"])
            ids.append(doc_id)
        
        # Reset index and ID mappings
        self._create_index()
        self.id_map = {}
        self.rev_id_map = {}
        self.next_id = 0
        
        # Re-add all documents
        self.add_texts(texts, metadatas, ids)
    
    def save(self, directory: Optional[str] = None) -> str:
        """Save the vector database to disk"""
        directory = directory or self.storage_dir
        os.makedirs(directory, exist_ok=True)
        
        # Save the FAISS index
        index_path = os.path.join(directory, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save metadata and ID mappings
        data = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "metadata": self.metadata,
            "id_map": self.id_map,
            "rev_id_map": self.rev_id_map,
            "next_id": self.next_id
        }
        
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(data, f)
        
        return directory
    
    @classmethod
    def load(cls, directory: str) -> 'VectorDB':
        """Load a vector database from disk"""
        # Load metadata and ID mappings
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(
            model_name=data["model_name"],
            dimension=data["dimension"],
            storage_dir=directory
        )
        
        # Load FAISS index
        index_path = os.path.join(directory, "faiss_index.bin")
        instance.index = faiss.read_index(index_path)
        
        # Restore metadata and ID mappings
        instance.metadata = data["metadata"]
        instance.id_map = data["id_map"]
        instance.rev_id_map = data["rev_id_map"]
        instance.next_id = data["next_id"]
        
        return instance
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        num_documents = len(self.metadata)
        total_size = 0
        if self.index:
            total_size = self.index.ntotal
        
        return {
            "num_documents": num_documents,
            "vectors_in_index": total_size,
            "model_name": self.model_name,
            "dimension": self.dimension
        }


def get_vector_db(model_name: str = "all-MiniLM-L6-v2", storage_dir: str = "./vector_storage") -> VectorDB:
    """Factory function to get a vector database instance"""
    return VectorDB(model_name, storage_dir=storage_dir)
