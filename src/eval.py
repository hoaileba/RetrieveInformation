import os
import argparse
import json
import time
from typing import List, Dict, Any, Tuple
from modules.bm25.bm25 import BM25
from modules.word2vec.word2vec import Word2Vec

def load_documents(data_path: str) -> List[str]:
    """
    Load documents from a file
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
        
    with open(data_path, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
        
    return documents

def load_queries(queries_path: str) -> List[str]:
    """
    Load queries from a file
    """
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Queries path not found: {queries_path}")
        
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
        
    return queries

def load_relevance_judgments(judgments_path: str) -> Dict[str, List[int]]:
    """
    Load relevance judgments from a file
    Format: query_id \t doc_id \t relevance_score
    """
    if not os.path.exists(judgments_path):
        raise FileNotFoundError(f"Judgments path not found: {judgments_path}")
        
    judgments = {}
    
    with open(judgments_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = parts[0]
                doc_id = int(parts[1])
                relevance = int(parts[2])
                
                if query_id not in judgments:
                    judgments[query_id] = []
                    
                judgments[query_id].append((doc_id, relevance))
                
    return judgments

def load_model(model_type: str, model_path: str) -> Any:
    """
    Load a trained model from disk
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
        
    if model_type == "bm25":
        model = BM25.load_model(model_path)
    elif model_type == "word2vec":
        model = Word2Vec.load_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model

def evaluate_model(model: Any, model_type: str, queries: List[str], 
                  documents: List[str], judgments: Dict[str, List[Tuple[int, int]]] = None,
                  top_k: int = 10) -> Dict[str, float]:
    """
    Evaluate a model on the given queries
    """
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mrr": 0.0,
        "avg_response_time": 0.0
    }
    
    total_queries = len(queries)
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_mrr = 0.0
    total_time = 0.0
    
    for i, query in enumerate(queries):
        # Measure response time
        start_time = time.time()
        
        # Search using the model
        if model_type == "bm25":
            results = model.search(query, top_k)
        elif model_type == "word2vec":
            results = model.search(query, top_k)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        
        # Extract document IDs from results
        result_doc_ids = [result["document_id"] for result in results]
        
        # Calculate metrics if judgments are available
        if judgments and str(i) in judgments:
            relevant_docs = [doc_id for doc_id, relevance in judgments[str(i)] if relevance > 0]
            
            # Calculate precision, recall, and F1
            if result_doc_ids:
                relevant_retrieved = sum(1 for doc_id in result_doc_ids if doc_id in relevant_docs)
                precision = relevant_retrieved / len(result_doc_ids)
                recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                
                # Calculate MRR
                for rank, doc_id in enumerate(result_doc_ids, 1):
                    if doc_id in relevant_docs:
                        total_mrr += 1.0 / rank
                        break
    
    # Calculate average metrics
    if total_queries > 0:
        metrics["precision"] = total_precision / total_queries
        metrics["recall"] = total_recall / total_queries
        metrics["f1"] = total_f1 / total_queries
        metrics["mrr"] = total_mrr / total_queries
        metrics["avg_response_time"] = total_time / total_queries
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate BM25 or Word2Vec model")
    parser.add_argument("--model_type", type=str, required=True, choices=["bm25", "word2vec"],
                        help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the document data file")
    parser.add_argument("--queries_path", type=str, required=True,
                        help="Path to the queries file")
    parser.add_argument("--judgments_path", type=str, default=None,
                        help="Path to the relevance judgments file (optional)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save evaluation results (optional)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top results to retrieve")
    
    args = parser.parse_args()
    
    # Load documents and queries
    documents = load_documents(args.data_path)
    queries = load_queries(args.queries_path)
    
    # Load relevance judgments if provided
    judgments = None
    if args.judgments_path:
        judgments = load_relevance_judgments(args.judgments_path)
    
    # Load model
    model = load_model(args.model_type, args.model_path)
    
    # Evaluate model
    metrics = evaluate_model(model, args.model_type, queries, documents, judgments, args.top_k)
    
    # Print results
    print(f"Evaluation results for {args.model_type} model:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results if output file is provided
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
