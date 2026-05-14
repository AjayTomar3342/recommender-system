import logging
import torch
from sentence_transformers import SentenceTransformer


def generate_embeddings(text_list, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)

    # Use Metal (MPS) for Mac GPU acceleration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    # batch_size=512 makes it much faster and cooler on the hardware
    embeddings = model.encode(
        text_list,
        batch_size=512,
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True
    )
    return embeddings