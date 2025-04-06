from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
import os
import json
from src.models import SearchRequest, SearchResponse, TrainingRequest
from modules.bm25.bm25 import BM25
# from modules.word2vec.word2vec import Word2Vec
from modules.huggingface_model.huggingface_model import HuggingFaceSearch
from modules.hybrid.hybrid_search import HybridSearch
from src.eval import evaluate_model, load_queries, load_relevance_judgments
import logging
import copy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

documents = []
with open("datasets/processed/processed_data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

mapping_disease_id_to_disease_name = {}
for disease in documents:
    mapping_disease_id_to_disease_name[disease["id"]] = disease

app = FastAPI()


# Model paths
MODEL_DIR = "weights"
BM25_MODEL_PATH = os.path.join(MODEL_DIR, "bm25/bm25_model.pkl")
EMBEDDING_PATH = os.path.join(MODEL_DIR, "huggingface/embeddings.pkl")

logger.info(f"Embedding path: {EMBEDDING_PATH}")

logger.info("Loading BM25 model")
bm25_model = BM25(documents=copy.deepcopy(documents))
bm25_model.fit()
logger.info("Loading HuggingFace model")
huggingface_model = HuggingFaceSearch(documents=copy.deepcopy(documents), embeddings_path=EMBEDDING_PATH)
logger.info("Loading Hybrid model")
hybrid_model = HybridSearch(documents=copy.deepcopy(documents), bm25=bm25_model, hf_search=huggingface_model, bm25_filter_k=30)

@app.post("/evaluate")
async def evaluate(request: Dict[str, Any]):
    """
    Evaluate a model on the given queries
    """
    model_type = request.get("model_type")
    queries_path = request.get("queries_path")
    judgments_path = request.get("judgments_path")
    top_k = request.get("top_k", 10)
    
    if not model_type:
        raise HTTPException(status_code=400, detail="Model type is required")
        
    if not queries_path:
        raise HTTPException(status_code=400, detail="Queries path is required")
        
    if not os.path.exists(queries_path):
        raise HTTPException(status_code=404, detail="Queries path not found")
        
    # Load queries
    queries = load_queries(queries_path)
    
    # Load relevance judgments if provided
    judgments = None
    if judgments_path:
        if not os.path.exists(judgments_path):
            raise HTTPException(status_code=404, detail="Judgments path not found")
        judgments = load_relevance_judgments(judgments_path)
    
    # Evaluate model
    if model_type == "bm25":
        metrics = evaluate_model(bm25_model, model_type, queries, documents, judgments, top_k)
    # elif model_type == "word2vec":
    #     metrics = evaluate_model(word2vec_model, model_type, queries, documents, judgments, top_k)
    elif model_type == "huggingface":
        metrics = evaluate_model(huggingface_model, model_type, queries, documents, judgments, top_k)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")
        
    return metrics

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search using the specified model
    """
    if request.model_type == "bm25":
        results = bm25_model.search(request.query, request.top_k)
    elif request.model_type == "huggingface":
        results = huggingface_model.search(request.query,request.top_k)
    elif request.model_type == "hybrid":
        results = hybrid_model.search(request.query, request.top_k)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    return SearchResponse(
        results=results['results'], 
        model_used=request.model_type, 
        query_time=results['timing']['query_time'], 
        number_of_results=len(results['results'])
    )

@app.get("/health")
def health():
    return {"message": "OK"}


@app.get("/disease/{disease_id}")
async def get_disease(disease_id: int):
    disease = mapping_disease_id_to_disease_name[disease_id]
    result = {
        "id": disease_id,
        "title": disease["name"],
        "content": disease["original_text"]
    }
    return result