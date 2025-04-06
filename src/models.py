from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    model_type: str  # 'bm25', 'word2vec', or 'huggingface'

class SearchResponse(BaseModel):
    model_used: str
    query_time: float
    number_of_results: int
    results: List[Dict[str, Any]]

class TrainingRequest(BaseModel):
    model_type: str
    data_path: str
    params: Optional[Dict[str, Any]] = None

