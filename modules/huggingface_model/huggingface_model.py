from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
import pickle
import os
import time
import logging



class HuggingFaceSearch:
    def __init__(self, model_name: str = "BAAI/bge-m3", documents: List[Dict[str, Any]] = [], embeddings_path: str = ""):
        print(f"Loading HuggingFace model {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model {model_name}")
        self.model = BGEM3FlagModel(model_name,  
                       use_fp16=True) 
        self.documents = documents
        self.processed_documents = []
        if os.path.exists(embeddings_path):
            print(f"Loading embeddings from {embeddings_path}")
            self.load_embeddings(embeddings_path)
        else:
            print(f"No embeddings found at {embeddings_path}, creating new embeddings")
            for doc in self.documents:
                self.processed_documents.append(doc.get('processed_text'))
            self.index(embeddings_path)

    def index(self, embeddings_path: str):
        # print(self.processed_documents)
        self.embeddings = self.get_embeddings(self.processed_documents)
        self.save_embeddings(self.embeddings, embeddings_path)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts
        """
        # Tokenize texts
        # print(texts)
        list_embeddings = []
        for text in tqdm(texts):
            embeddings = self.model.encode([text], 
                                )['dense_vecs']
            list_embeddings.append(embeddings)
        return list_embeddings
        
    def load_embeddings(self, embeddings_path: str):
        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)
        self.embeddings = np.squeeze(self.embeddings, axis=1)


    def save_embeddings(self, embeddings: List[np.ndarray], embeddings_path: str):
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)


    def search(self, query: str, top_k: int = 10, list_ids: List[int] = None) -> Dict[str, Any]:
        """
        Search for documents using HuggingFace model
        
        Args:
            query: Search query
            top_k: Number of results to return
            list_ids: Optional list of document IDs to search within
            
        Returns:
            Dictionary containing:
            - results: List of documents with scores
            - timing: Dictionary with timing information
        """
        start_time = time.time()
        
        # Get query embedding
        embedding_start = time.time()
        query_embedding = self.get_embeddings([query])[0]
        embedding_time = time.time() - embedding_start
        
        # Get document embeddings
        if list_ids is None:    
            doc_embeddings = self.embeddings
        else:
            doc_embeddings = self.embeddings[list_ids]
        
        # Calculate cosine similarity
        similarity_start = time.time()
        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
        similarity_time = time.time() - similarity_start
        
        # Get top k results
        ranking_start = time.time()
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        # print("documents: ", len(self.documents))
        results = []
        for idx in top_k_indices:

            # print("idx: ", idx)
            # print("self.documents[idx]: ", self.documents[idx])
            title = self.documents[idx].get("name", "")
            if title == "":
                title = self.documents[idx].get("title", "")
            
            description = self.documents[idx].get("processed_text", "")
            if description == "":
                description = self.documents[idx].get("description", "")
            
            results.append({
                "id": self.documents[idx]["id"],
                "title": title,
                "description": description[:200] + "..."
            })
        ranking_time = time.time() - ranking_start
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "timing": {
                "embedding_time": round(embedding_time, 4),
                "similarity_time": round(similarity_time, 4),
                "ranking_time": round(ranking_time, 4),
                "query_time": round(total_time, 4)
            }
        }