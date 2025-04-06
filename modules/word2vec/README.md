# Word2Vec Implementation

This module provides a Word2Vec implementation with the following features:

- Skip-gram model with softmax loss
- Sentence embedding generation
- Document search using word embeddings

## Usage

### Basic Usage

```python
from word2vec import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Your corpus of documents
corpus = [
    "Document 1 text here.",
    "Document 2 text here.",
    # ... more documents
]

# Initialize the model
model = Word2Vec(
    vector_size=100,  # Dimension of word vectors
    window=2,         # Context window size
    epochs=100,       # Number of training epochs
    learning_rate=0.007,  # Learning rate
    documents=corpus  # Your corpus
)

# Train the model
model.train()

# Get sentence vectors for visualization
sentence_vectors = model.get_sentence_vectors()

# Visualize sentence embeddings using PCA
pca = PCA(n_components=2)
sentence_vectors_2d = pca.fit_transform(sentence_vectors)

plt.figure(figsize=(10, 8))
plt.scatter(sentence_vectors_2d[:, 0], sentence_vectors_2d[:, 1], c='blue', edgecolors='k')
for i, (x, y) in enumerate(sentence_vectors_2d):
    plt.text(x + 0.001, y + 0.001, corpus[i][:12] + "...", fontsize=12)
plt.title("Sentence Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Search for similar documents
results = model.search("query text", top_k=5)
for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")
    print()

# Save the model
model.save_model("word2vec_model.pkl")

# Load a saved model
loaded_model = Word2Vec.load_model("word2vec_model.pkl")
```

### Example with Vietnamese Corpus

See `example.py` for a complete example using a Vietnamese corpus.

## Implementation Details

This implementation uses the Skip-gram model with softmax loss. The training process:

1. Builds a vocabulary from the provided documents
2. Generates training pairs (target word, context word)
3. Trains the model using gradient descent with the following steps:
   - Forward pass: compute softmax probabilities
   - Backward pass: compute gradients and update weights
4. Provides methods for document search

## Visualization

To visualize sentence embeddings, you can use the `get_sentence_vectors()` method to obtain the sentence vectors, then use PCA to reduce the dimensionality to 2D for visualization. See the example code above.

## Document Search

The `search()` method allows you to find documents similar to a query by:
1. Converting the query to a vector (average of word vectors)
2. Computing cosine similarity with each document vector
3. Returning the top-k most similar documents 