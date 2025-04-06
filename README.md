# Medical Information Retrieval System

## Project Structure

```
NLP_btl/
├── datasets/
│   ├── corpus/           # Raw medical documents
│   ├── processed/        # Processed documents
│   └── queries.json      # Test queries
├── modules/
│   ├── bm25/             # BM25 implementation
│   ├── word2vec/         # Word2Vec implementation
│   └── huggingface_model/ # HuggingFace model implementation
├── src/
│   ├── main.py           # FastAPI application
│   ├── train.py          # Training scripts
│   ├── eval.py           # Evaluation scripts
│   ├── preprocess.py     # Data preprocessing
│   └── models.py         # Data models
└── requirements.txt      # Project dependencies
```

## Installation

### Docker

1. Build the Docker image:

```bash
docker build -t nlp .
```

2. Up service:

```bash
docker compose up 
```

Note: It will take few minute to start because we have to load weight model BGE, it will depend on your internet's speed

## Data Preprocessing

The `preprocess.py` script processes the raw medical documents by:
1. Removing HTML tags
2. Extracting URLs
3. Formatting the data as JSON objects with the following keys:
   - `url`: The source URL of the document
   - `original_text`: The original text with HTML tags
   - `processed_text`: The cleaned text without HTML tags

### Usage

```bash
python -m src.preprocess --input_dir datasets/corpus --output_file datasets/processed/processed_data.json
```

## API Usage

After build docker image and start container you can access in http://localhost:8000/docs to test search or you can use below script to query

The FastAPI application provides endpoints for training, loading, evaluating, and searching:

### Search

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "triệu chứng bệnh viêm khớp",
       "model_type": "bm25",
       "top_k": 10
     }'
```

