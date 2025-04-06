from typing import List, Dict, Any
import numpy as np
import time
from ..bm25.bm25 import BM25
from ..huggingface_model.huggingface_model import HuggingFaceSearch

class HybridSearch:
    def __init__(self, 
                 documents: List[Dict[str, Any]], 
                 bm25: BM25,
                 hf_search: HuggingFaceSearch,  
                 bm25_filter_k: int = 20):
        """
        Initialize hybrid search combining BM25 and HuggingFace models
        
        Args:
            documents: List of documents to search through
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            hf_model_name: HuggingFace model name
            embeddings_path: Path to save/load embeddings
            bm25_filter_k: Number of documents to filter using BM25 before using HuggingFace
        """
        self.documents = documents
        self.bm25 = bm25
        self.hf_search = hf_search
        self.bm25_filter_k = bm25_filter_k
        
    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Hybrid search combining BM25 and HuggingFace models
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Dictionary containing:
            - results: List of documents with scores
            - timing: Dictionary with timing information for each step
        """
        # Step 1: Use BM25 to filter documents
        bm25_start_time = time.time()
        bm25_results = self.bm25.search(query, self.bm25_filter_k)
        bm25_time = time.time() - bm25_start_time
        
        # Step 2: Use HuggingFace model on filtered documents
        filtered_docs = [doc for doc in bm25_results['results']]
        list_ids = [doc['id'] for doc in filtered_docs]
        print("list_ids: ", list_ids)

        hf_start_time = time.time()
        # self.hf_search.documents = filtered_docs
        hf_results = self.hf_search.search(query, top_k, list_ids)
        hf_time = time.time() - hf_start_time
        
        # Calculate total time
        total_time = bm25_time + hf_time
        
        return {
            "results": hf_results['results'],
            "timing": {
                "bm25_time": round(bm25_time, 4),  # Time taken by BM25 filtering
                "huggingface_time": round(hf_time, 4),  # Time taken by HuggingFace search
                "query_time": round(total_time, 4)  # Total time taken
            }
        } 