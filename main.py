import yaml
import os
import logging
import sqlite3
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
import time
from sklearn.preprocessing import LabelEncoder

# Your Existing Modules
from src.ingestion.uploader import DataIngestor
from src.transformation.transformer import DBTRunner
from models.nlp.generate_embeddings import generate_embeddings
from models.vector_store.chroma_manager import save_to_chroma

# NCF Imports
from src.models.architecture import NCF
from src.features.dataloader import NCFDataset, ChromaAccessor
from src.models.train import train_model
from src.models.evaluate import evaluate_model

# 1. Logging Setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "pipeline_execution.log"),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Orchestrator")


def fetch_data_from_db(db_path, sample_rate=0.01):
    """Fetches data from SQLite."""
    print(f"🔍 Connecting to database at: {db_path}")
    if not os.path.exists(db_path):
        print(f" DATABASE NOT FOUND at {db_path}")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT * FROM fact_reviews 
        LIMIT (SELECT CAST(COUNT(*) * {sample_rate} AS INT) FROM fact_reviews)
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f" SQL Error: {e}")
        if conn: conn.close()
        return pd.DataFrame()


def main():
    # --- MLFLOW EXPERIMENT SETUP ---
    mlflow.set_experiment("Amazon_Recommender_Project")

    with mlflow.start_run(run_name="Full_Pipeline_Run"):
        start_time = time.time()

        print("\n" + "=" * 40)
        print(" 🚀 PIPELINE START: FULL MLFLOW INTEGRATION")
        print("=" * 40)

        # Load Config
        project_root = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(project_root, "config", "base_config.yaml")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 1. LOG PARAMETERS & ARTIFACTS
        mlflow.log_artifact(config_path, artifact_path="config")
        mlflow.log_param("sample_rate", config.get("sampling", {}).get("rate", 0.01))
        mlflow.log_param("epochs_total", config.get("training", {}).get("epochs", 5))
        mlflow.log_param("device", "mps" if torch.backends.mps.is_available() else "cpu")

        # --- DATA FETCHING & PREPROCESSING ---
        db_rel_path = config["local_db"]["path"]
        db_path = os.path.join(project_root, db_rel_path)
        sample_rate = config.get("sampling", {}).get("rate", 0.01)

        df = fetch_data_from_db(db_path, sample_rate=sample_rate)

        if not df.empty:
            df.rename(columns={'product_id': 'item_id'}, inplace=True)

            initial_count = len(df)
            df = df.drop_duplicates(subset=['review_pk'], keep='first')
            final_count = len(df)

            # Calculate Data Metrics
            num_users = df['user_id'].nunique()
            num_items = df['item_id'].nunique()
            sparsity = 1.0 - (len(df) / (num_users * num_items))

            mlflow.log_metric("initial_rows", initial_count)
            mlflow.log_metric("cleaned_rows", final_count)
            mlflow.log_metric("unique_users", num_users)
            mlflow.log_metric("unique_items", num_items)
            mlflow.log_metric("data_sparsity", round(sparsity, 4))

            # ID ENCODING
            user_encoder = LabelEncoder()
            item_encoder = LabelEncoder()
            df['user_id'] = user_encoder.fit_transform(df['user_id'].astype(str))
            df['item_id'] = item_encoder.fit_transform(df['item_id'].astype(str))
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)

            # --- PHASE 3 & 4: NLP & VECTOR ---
            texts = df['review_body'].tolist()
            ids = df['review_pk'].astype(str).tolist()

            print(f" Phase 3: Generating Embeddings...")
            vectors = generate_embeddings(texts)

            print(" Phase 4: Saving to ChromaDB...")
            save_to_chroma(texts=texts, embeddings=vectors.tolist(), ids=ids)

            # --- PHASE 5: NCF TRAINING ---
            try:
                print(" 🤖 Phase 5: Training NCF Model...")
                vector_store_path = os.path.join(project_root, "data", "vector_store")
                accessor = ChromaAccessor(db_path=vector_store_path)

                dataset = NCFDataset(df, vector_store=accessor)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

                model = NCF(num_users=num_users, num_items=num_items)

                # Log model size
                total_params = sum(p.numel() for p in model.parameters())
                mlflow.log_param("model_total_parameters", total_params)

                # Training with metric per epoch
                epochs = config.get("training", {}).get("epochs", 5)
                for epoch in range(epochs):
                    # Calling train_model for 1 epoch at a time to log progress
                    avg_loss = train_model(model, dataloader, epochs=1)

                    # --- THE FIX: Convert Tensor to Float for MLflow ---
                    if hasattr(avg_loss, 'item'):
                        avg_loss = avg_loss.item()
                    elif isinstance(avg_loss, (list, tuple)):
                        avg_loss = float(avg_loss[0])
                    else:
                        avg_loss = float(avg_loss) if avg_loss is not None else 0.0

                    mlflow.log_metric("train_loss", avg_loss, step=epoch)
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

                # --- PHASE 6: EVALUATION ---
                print(" 🔍 Phase 6: Evaluating Model...")
                accuracy = evaluate_model(model, dataloader)

                # Final logs
                duration = time.time() - start_time
                mlflow.log_metric("final_accuracy", accuracy)
                mlflow.log_metric("pipeline_duration_sec", round(duration, 2))

                # Registering model in MLflow
                mlflow.pytorch.log_model(model, artifact_path="ncf_model")

                print(f"\n" + "=" * 40)
                print(f"📊 FINAL MODEL ACCURACY: {accuracy:.2f}%")
                print("=" * 40)

            except Exception as e:
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error_message", str(e))
                print(f"❌ Error during training/eval: {e}")
                import traceback
                traceback.print_exc()

            print("\n✅ PIPELINE COMPLETE. Model and Metrics tracked.")
        else:
            print("🛑 Pipeline stopped: No data retrieved.")


if __name__ == "__main__":
    main()

# uvicorn app:app --reload
# http://127.0.0.1:8000/docs

# {
#   "user_id": 0,
#   "item_id": 0,
#   "review_pk": "ballsy"
# }

# docker-compose -f docker/docker-compose.yml up --build
