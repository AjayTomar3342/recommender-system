import torch
from torch.utils.data import Dataset
import chromadb


class ChromaAccessor:
    def __init__(self, db_path, collection_name="product_reviews"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

    def get_embedding(self, item_id):
        # Retrieve the embedding for a specific ID
        results = self.collection.get(ids=[str(item_id)], include=['embeddings'])
        embeddings = results.get('embeddings')

        # Check if embeddings exists and contains at least one item
        # We check "is not None" and "len > 0" specifically to avoid the truth-value error
        if embeddings is not None and len(embeddings) > 0:
            return embeddings[0]

        # Return a zero-vector if not found (fallback)
        return [0.0] * 384


class NCFDataset(Dataset):
    def __init__(self, df, vector_store):
        """
        df: DataFrame containing user_id, item_id, rating
        vector_store: Accessor for your ChromaDB embeddings
        """
        # Explicitly casting values to int before tensor conversion
        self.users = torch.tensor(df['user_id'].values.astype(int), dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values.astype(int), dtype=torch.long)

        # Normalize ratings from [1, 5] range to [0, 1] range for Binary Cross Entropy
        # Formula: (rating - 1) / 4
        self.labels = torch.tensor(((df['rating'].values.astype(float) - 1) / 4), dtype=torch.float32)

        self.vector_store = vector_store

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        item_id = self.items[idx]
        label = self.labels[idx]

        # Retrieve the 'vibe' vector from ChromaDB
        vec = self.vector_store.get_embedding(item_id.item())
        review_vector = torch.tensor(vec, dtype=torch.float32)

        return user_id, item_id, review_vector, label