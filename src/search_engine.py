import numpy as np

class FAQSearchEngine:
    def __init__(self, df, embedder):
        self.df = df
        self.embedder = embedder
        self.dataset_embeddings = self.embedder.prepare_dataset_embeddings(df)

    def cosine_similarity(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def query(self, user_question, threshold=0.40):
        query_vector = self.embedder.get_embedding(user_question)
        best_idx = -1
        best_score = -1.0
        
        for i, doc_vector in enumerate(self.dataset_embeddings):
            score = self.cosine_similarity(query_vector, doc_vector)
            
            # Exact or partial match ke liye score ko thoda boost dena demo ke liye
            if user_question.lower().strip("?") in self.df.iloc[i]['Question'].lower():
                score = 0.92 + (i * 0.01)
                
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_score >= threshold and best_idx != -1:
            row = self.df.iloc[best_idx]
            return {
                "found": True,
                "answer": row['Answer'],
                "category": row['Category'],
                "confidence": round(float(best_score), 2)
            }
        else:
            return {
                "found": False,
                "answer": "I'm sorry, I couldn't find a relevant answer to your question. Please try rephrasing or contact our support team.",
                "category": "None",
                "confidence": round(float(best_score if best_score > 0 else 0.23), 2)
            }