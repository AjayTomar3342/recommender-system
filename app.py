import os
import torch
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import your Chroma accessor to fetch the missing vectors
from src.features.dataloader import ChromaAccessor

# --- CONFIGURATION ---
RUN_ID = "1421a87893b74f9ab65b9f219417e8f4"
MODEL_URI = f"runs:/{RUN_ID}/ncf_model"
VECTOR_STORE_PATH = os.path.join(os.getcwd(), "data", "vector_store")

app = FastAPI(title="Amazon Recommender API")

model = None
accessor = None


@app.on_event("startup")
def load_resources():
    global model, accessor
    try:
        # 1. Load Model
        model = mlflow.pytorch.load_model(MODEL_URI)
        model.eval()

        # 2. Load Vector Store Accessor
        accessor = ChromaAccessor(db_path=VECTOR_STORE_PATH)
        print("✅ Model and Vector Store loaded.")
    except Exception as e:
        print(f"❌ Startup Error: {e}")


class RecommendRequest(BaseModel):
    user_id: int
    item_id: int
    review_pk: str  # We need this to find the specific vector in Chroma


@app.post("/predict")
def predict(request: RecommendRequest):
    if model is None or accessor is None:
        raise HTTPException(status_code=503, detail="Resources not ready")

    try:
        # 1. Fetch the embedding from ChromaDB using the review_pk
        # This matches the 'review_vectors' argument the model is asking for
        vector = accessor.get_embedding(request.review_pk)

        if vector is None:
            # Fallback to zeros if not found, or raise error
            vector = torch.zeros(384)
        else:
            vector = torch.tensor(vector)

        # 2. Prepare Tensors
        user_t = torch.LongTensor([request.user_id])
        item_t = torch.LongTensor([request.item_id])
        vector_t = vector.unsqueeze(0)  # Add batch dimension

        # 3. Inference
        with torch.no_grad():
            # Now passing all 3 required arguments
            prediction = model(user_t, item_t, vector_t)
            score = float(prediction.item())

        return {
            "user_id": request.user_id,
            "item_id": request.item_id,
            "score": round(score, 4)
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))