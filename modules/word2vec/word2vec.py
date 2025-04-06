import numpy as np
from typing import List, Dict, Any
import pickle
import os
from tqdm import tqdm

class Word2Vec:
    def __init__(self, 
                 vector_size: int = 100,
                 window: int = 2,
                 min_count: int = 1,
                 workers: int = 4,
                 epochs: int = 100,
                 learning_rate: float = 0.007,
                 documents: List[str] = None):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.documents = documents
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_vectors = None
        self.context_vectors = None
        self.vocab_size = 0
        self.training_pairs = []
        
    def save_model(self, path: str):
        """
        Save the trained model to disk
        """
        model_data = {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'word_vectors': self.word_vectors,
            'context_vectors': self.context_vectors,
            'vocab_size': self.vocab_size
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod
    def load_model(cls, path: str) -> 'Word2Vec':
        """
        Load a trained model from disk
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        # Create new instance
        model = cls(
            vector_size=model_data['vector_size'],
            window=model_data['window'],
            min_count=model_data['min_count']
        )
        
        # Load model data
        model.word_to_index = model_data['word_to_index']
        model.index_to_word = model_data['index_to_word']
        model.word_vectors = model_data['word_vectors']
        model.context_vectors = model_data['context_vectors']
        model.vocab_size = model_data['vocab_size']
        
        return model
    
    def train(self):
        """
        Train Word2Vec model using Skip-gram with softmax
        """
        # 1. Tách câu thành token (dựa vào dấu cách)
        tokenized_corpus = [sentence.split() for sentence in self.documents]
        
        # 2. Xây dựng từ điển
        words = [word for sentence in tokenized_corpus for word in sentence]
        vocab = list(set(words))
        self.vocab_size = len(vocab)
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(vocab)}
        
        # 3. Sinh các cặp training (skip-gram)
        self.training_pairs = []
        for sentence in tokenized_corpus:
            sentence_indices = [self.word_to_index[word] for word in sentence]
            for i, target in enumerate(sentence_indices):
                # Lấy các từ ngữ cảnh trong cửa sổ, loại bỏ từ chính (target)
                for j in range(max(0, i - self.window), min(len(sentence_indices), i + self.window + 1)):
                    if j != i:
                        context = sentence_indices[j]
                        self.training_pairs.append((target, context))
        
        # 4. Khởi tạo trọng số cho mô hình
        # Ma trận embedding (W): kích thước (vocab_size x embedding_size)
        self.word_vectors = np.random.randn(self.vocab_size, self.vector_size) * 0.01
        
        # Ma trận đầu ra (W'): kích thước (embedding_size x vocab_size)
        self.context_vectors = np.random.randn(self.vector_size, self.vocab_size) * 0.01
        
        # 5. Huấn luyện mô hình
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0
            # Xáo trộn các cặp training
            np.random.shuffle(self.training_pairs)
            for target_idx, context_idx in tqdm(self.training_pairs):
                # --- Forward ---
                # h: hidden_state
                h = self.word_vectors[target_idx]  # shape: (embedding_size,)
                # Tính đầu ra: u = h^T * W_prime (kích thước vocab_size)
                u = np.dot(h, self.context_vectors)  # shape: (vocab_size,)
                y_pred = self.softmax(u)
                
                # Hàm mất mát: -log(prob(context))
                loss = -np.log(y_pred[context_idx] + 1e-9)
                total_loss += loss
                
                # --- Backward ---
                # Đạo hàm của loss theo u: y_pred - y_true, với y_true là vector one-hot
                y_true = np.zeros(self.vocab_size)
                y_true[context_idx] = 1
                # gradient
                e = y_pred - y_true  # shape: (vocab_size,)
                
                # Đạo hàm theo W_prime: outer product của h và e
                grad_W_prime = np.outer(h, e)  # shape: (embedding_size, vocab_size)
                
                # Đạo hàm theo h (các tham số của W đối với từ target)
                grad_h = np.dot(self.context_vectors, e)   # shape: (embedding_size,)
                
                # Cập nhật trọng số (chỉ cập nhật cho từ target)
                self.word_vectors[target_idx] -= self.learning_rate * grad_h
                self.context_vectors -= self.learning_rate * grad_W_prime
                
            avg_loss = total_loss / len(self.training_pairs)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax values for each set of scores in x
        """
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex)
    
    def get_sentence_vectors(self) -> np.ndarray:
        """
        Generate sentence vectors by averaging word vectors
        """
        # Tạo vector đại diện cho mỗi câu
        sentence_vectors = []
        tokenized_corpus = [sentence.split() for sentence in self.documents]
        
        for sentence in tokenized_corpus:
            word_vecs = [self.word_vectors[self.word_to_index[word]] for word in sentence 
                        if word in self.word_to_index]
            
            if word_vecs:
                sentence_vector = np.mean(word_vecs, axis=0)
                sentence_vectors.append(sentence_vector)
            else:
                # If no valid words, use zero vector
                sentence_vectors.append(np.zeros(self.vector_size))
                
        return np.array(sentence_vectors)
    
            
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents using word embeddings
        """
        # Preprocess query
        query_words = query.split()
        
        # Get query vector (average of word vectors)
        query_vec = np.zeros(self.vector_size)
        valid_words = 0

        for word in query_words:
            if word in self.word_to_index:
                query_vec += self.word_vectors[self.word_to_index[word]]
                valid_words += 1
                
        if valid_words == 0:
            return []
            
        query_vec /= valid_words
        
        # Calculate similarity scores
        scores = []
        for i, doc in enumerate(self.documents):
            doc_words = doc.split()
            doc_vec = np.zeros(self.vector_size)
            valid_words = 0
            
            for word in doc_words:
                if word in self.word_to_index:
                    doc_vec += self.word_vectors[self.word_to_index[word]]
                    valid_words += 1
                    
            if valid_words == 0:
                continue
                
            doc_vec /= valid_words
            
            # Calculate cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            
            scores.append((i, similarity))
            
        # Sort by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for doc_idx, score in scores[:top_k]:
            results.append({
                "document_id": doc_idx,
                "score": float(score),
                "text": self.documents[doc_idx]
            })
            
        return results
