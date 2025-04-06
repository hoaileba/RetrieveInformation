import json
import re
import math
import numpy as np
import string
import time
from typing import List, Dict, Any

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75, documents: List[str] = []):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.corpus_text = []
        for doc in self.documents:
            self.corpus_text.append(doc.get('processed_text'))
        
    def clean_text(self, text):
        text = re.sub('<.*?>', '', text).strip()
        text = re.sub('(\s)+', r'\1', text)
        return text
        
    def normalize_text(self, text):
        listpunctuation = string.punctuation.replace('_', '')
        for i in listpunctuation:
            text = text.replace(i, ' ')
        return text.lower()
    
    def word_segment(self, text):
        return text.strip().split(" ")
    
    def pre_process(self, text):
        text_cleaned = self.clean_text(text)
        text_cleaned = self.normalize_text(text_cleaned)
        text_segmented = self.word_segment(text)
        # print(text_segmented)
        return text_segmented
        
    def calculate_tf_idf(self, corpus):
        tf = []
        df = {}
        idf = {}
        len_corpus = len(corpus)
        for document in corpus:
            freq = {}
            #  Tính tf
            for term in document:
                term_count = freq.get(term, 0) + 1
                freq[term] = term_count
            tf.append(freq)
            # Tính df
            for term, term_count in freq.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count
            
        # tính idf
        for term, freq_doc in df.items():
            idf[term] = math.log(
                1 + (len_corpus - freq_doc + 0.5)/(freq_doc + 0.5)
            )
        return tf, df, idf
        
        
    def fit(self):
        
        self.corpus = [self.pre_process(doc) for doc in self.corpus_text]
        tf, df, idf = self.calculate_tf_idf(self.corpus)
        self.tf = tf
        self.df = df
        self.idf = idf
        self.len_corpus = len(self.corpus)
        doc_len = []
        for doc in self.corpus:
            doc_len.append(len(doc))
        self.doc_len = doc_len
        self.avg_doc_len = sum(doc_len)/self.len_corpus
        
        return self
    
    def scoring_query_doc(self,query, index):
        score = 0
        doc_len = self.doc_len[index]
        freq = self.tf[index]
        for term in query:
            if term in freq:
                freq_term = freq.get(term)
                score += self.idf[term] * freq_term * (self.k1 + 1) / (freq_term + self.k1 * (1 - self.b + self.b * (doc_len/self.avg_doc_len)))
        return score    
    
    def ranking(self, query: str) -> list:
        query_segment = self.pre_process(query)
        scores = [self.scoring_query_doc(
            query=query_segment,
            index= index
        ) for index in range(self.len_corpus)]
        return scores
    
    def search(self, query, top_k) -> Dict[str, Any]:
        """
        Search for documents using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Dictionary containing:
            - results: List of documents with scores
            - timing: Dictionary with timing information
        """
        start_time = time.time()
        
        # Pre-process query
        preprocess_start = time.time()
        query_segment = self.pre_process(query)
        preprocess_time = time.time() - preprocess_start
        
        # Calculate scores
        scoring_start = time.time()
        scores = [self.scoring_query_doc(
            query=query_segment,
            index= index
        ) for index in range(self.len_corpus)]
        scoring_time = time.time() - scoring_start
        
        # Sort and get top k results
        sorting_start = time.time()
        scores_index = np.argsort(scores)
        scores_index = scores_index[::-1]
        if top_k > 0:
            results = []
            for index in scores_index[:top_k]:
                results.append(self.documents[index])
        else:
            results = []
            for index in scores_index:
                results.append(self.documents[index])
        sorting_time = time.time() - sorting_start
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "timing": {
                "preprocess_time": round(preprocess_time, 4),
                "scoring_time": round(scoring_time, 4),
                "sorting_time": round(sorting_time, 4),
                "total_time": round(total_time, 4)
            }
        }
        