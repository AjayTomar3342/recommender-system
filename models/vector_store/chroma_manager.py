import chromadb
import os
import logging

logger = logging.getLogger(__name__)


def save_to_chroma(texts, embeddings, ids):
    """
    Saves text and embeddings to a local persistent ChromaDB.
    Chunks the data to stay below Chroma's internal batch limits.
    """
    # 1. Setup Path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    db_path = os.path.join(project_root, "data", "vector_store")

    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)

    # 2. Initialize Client
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="product_reviews")

    # 3. CHUNKING LOGIC
    batch_size = 5000
    total_items = len(ids)

    print(f" Saving {total_items} items to ChromaDB in chunks...")

    for i in range(0, total_items, batch_size):
        end_idx = min(i + batch_size, total_items)

        batch_texts = texts[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_ids = ids[i:end_idx]

        collection.upsert(
            documents=batch_texts,
            embeddings=batch_embeddings,
            ids=batch_ids
        )
        print(f" Chunk {i // batch_size + 1}: Items {i} to {end_idx} saved.")

    print(f" Successfully persisted all {total_items} items to: {db_path}")