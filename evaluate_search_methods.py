import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from modules.bm25.bm25 import BM25
from modules.huggingface_model.huggingface_model import HuggingFaceSearch
from modules.hybrid.hybrid_search import HybridSearch
import copy
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load documents
documents = []
with open("datasets/processed/processed_data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load queries
queries = []
with open("datasets/queries.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        queries.append(json.loads(line))

# Load relevance judgments
with open("datasets/label.json", "r", encoding="utf-8") as f:
    relevance_judgments_raw = json.load(f)

relevance_judgments = {}
for query_id, docs in relevance_judgments_raw.items():
    list_relevant_docs = []
    for doc in docs:
        list_relevant_docs.append(doc.split("|")[0])
    list_relevant_docs = list(set(list_relevant_docs))
    relevance_judgments[query_id] = list_relevant_docs

# Model paths
MODEL_DIR = "weights"
BM25_MODEL_PATH = os.path.join(MODEL_DIR, "bm25/bm25_model.pkl")
EMBEDDING_PATH = os.path.join(MODEL_DIR, "huggingface/embeddings.pkl")

# Load models
logger.info("Loading BM25 model")
bm25_model = BM25(documents=copy.deepcopy(documents))
bm25_model.fit()

logger.info("Loading HuggingFace model")
huggingface_model = HuggingFaceSearch(documents=copy.deepcopy(documents), embeddings_path=EMBEDDING_PATH)

logger.info("Loading Hybrid model")
hybrid_model = HybridSearch(documents=copy.deepcopy(documents), bm25=bm25_model, hf_search=huggingface_model, bm25_filter_k=30)

def calculate_recall_at_k(relevant_docs: List[str], retrieved_docs: List[Dict[str, Any]], k: int) -> float:
    """Calculate recall@k"""
    if not relevant_docs:
        return 0.0
    
    retrieved_at_k = retrieved_docs[:k]
    retrieved_ids = [doc["title"] for doc in retrieved_at_k]
    relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_docs)
    return relevant_retrieved / len(relevant_docs)

def calculate_mrr_at_k(relevant_docs: List[str], retrieved_docs: List[Dict[str, Any]], k: int) -> float:
    """Calculate MRR@k"""
    if not relevant_docs:
        return 0.0
    
    retrieved_at_k = retrieved_docs[:k]
    for rank, doc in enumerate(retrieved_at_k, 1):
        if doc["title"] in relevant_docs:
            return 1.0 / rank
    return 0.0

def evaluate_search_method(model, model_name: str, queries: List[Dict[str, Any]], relevance_judgments: Dict[str, List[str]], top_k: int = 5) -> Dict[str, float]:
    """
    Evaluate a search method on the given queries
    
    Args:
        model: The search model to evaluate
        model_name: Name of the model for logging
        queries: List of queries to evaluate
        relevance_judgments: Dictionary mapping query IDs to lists of relevant document IDs
        top_k: Maximum number of results to retrieve
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name} model")
    # print(f"relevance_judgments: {relevance_judgments}")
    # return
    total_recall_at_3 = 0.0
    total_recall_at_5 = 0.0
    total_mrr_at_3 = 0.0
    total_mrr_at_5 = 0.0
    total_time = 0.0
    
    for query_data in tqdm(queries, desc=f"Evaluating {model_name}"):
        query_id = str(query_data["query_id"])
        query_text = query_data["query"]
        # print(f"query_text: {query_text}")
        # Get relevant documents for this query
        if query_id not in relevance_judgments:
            logger.warning(f"No relevance judgments for query {query_id}")
            continue
            
        relevant_docs_for_query = relevance_judgments[query_id]
        
        # Run search and measure time
        start_time = time.time()
        results = model.search(query_text, top_k)
        # print(f"results: {results}")
        # return
        query_time = time.time() - start_time
        total_time += query_time
        
        # Calculate metrics
        recall_at_3 = calculate_recall_at_k(relevant_docs_for_query, results["results"], 3)
        recall_at_5 = calculate_recall_at_k(relevant_docs_for_query, results["results"], 5)
        mrr_at_3 = calculate_mrr_at_k(relevant_docs_for_query, results["results"], 3)
        mrr_at_5 = calculate_mrr_at_k(relevant_docs_for_query, results["results"], 5)
        
        total_recall_at_3 += recall_at_3
        total_recall_at_5 += recall_at_5
        total_mrr_at_3 += mrr_at_3
        total_mrr_at_5 += mrr_at_5
        
        logger.debug(f"Query {query_id}: Recall@3={recall_at_3:.4f}, Recall@5={recall_at_5:.4f}, "
                   f"MRR@3={mrr_at_3:.4f}, MRR@5={mrr_at_5:.4f}, Time={query_time:.4f}s")
    
    # Calculate average metrics
    num_queries = len(queries)
    metrics = {
        "recall@3": total_recall_at_3 / num_queries,
        "recall@5": total_recall_at_5 / num_queries,
        "mrr@3": total_mrr_at_3 / num_queries,
        "mrr@5": total_mrr_at_5 / num_queries,
        "avg_query_time": total_time / num_queries,
        "total_time": total_time
    }
    
    return metrics

def main():
    # Evaluate each model
    models = {
        "BM25": bm25_model,
        "HuggingFace": huggingface_model,
        "Hybrid": hybrid_model
    }
    
    results = {}
    for model_name, model in models.items():
        metrics = evaluate_search_method(model, model_name, queries, relevance_judgments)
        results[model_name] = metrics
    
    # Print comparison table
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"{'Model':<15} {'Recall@3':<10} {'Recall@5':<10} {'MRR@3':<10} {'MRR@5':<10} {'Avg Time (s)':<15} {'Total Time (s)':<15}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['recall@3']:<10.4f} {metrics['recall@5']:<10.4f} "
              f"{metrics['mrr@3']:<10.4f} {metrics['mrr@5']:<10.4f} "
              f"{metrics['avg_query_time']:<15.4f} {metrics['total_time']:<15.4f}")
    
    print("-" * 80)
    
    # Save results to file
    output_file = "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()