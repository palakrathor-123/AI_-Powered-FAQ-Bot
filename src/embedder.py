import numpy as np
import pandas as pd

class FAQEmbedder:
    def __init__(self):
        # Production mein aap yahan model load karenge:
        # self.model = SentenceTransformer('all-MiniLM-L6-v2')
        pass

    def get_embedding(self, text):
        # Vector embedding generation ka mockup logic
        np.random.seed(hash(text) % (2**32 - 1))
        return np.random.rand(384)

    def prepare_dataset_embeddings(self, df):
        # Dataset ke saare sawalon ke liye embeddings generate karna
        embeddings = []
        for q in df['Question']:
            embeddings.append(self.get_embedding(q))
        return np.array(embeddings)