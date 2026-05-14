import json
import sqlite3
import os
from typing import Dict, Any, Optional
from src.utils.logger import setup_custom_logger

logger = setup_custom_logger("IngestionModule")


class DataIngestor:
    """
    Requirement A: Class-based modularity.
    Requirement B: Type Hinting & Docstrings.
    """

    def __init__(self, config: Dict[str, Any]):
        self.db_path = config["local_db"]["path"]
        self.table_name = config["local_db"]["table_name"]
        self.params = config["ingestion_params"]
        self.conn: Optional[sqlite3.Connection] = None

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        logger.info(f"DataIngestor initialized for SQLite database.")

    def connect(self) -> None:
        """Establishes connection and creates raw table."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    RAW_CONTENT TEXT, 
                    INGESTED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
            logger.info(f"Connected to {self.db_path}. Table {self.table_name} verified.")
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Security B: Validate mandatory keys."""
        keys = self.params["mandatory_keys"]
        return all(key in record for key in keys)

    def run_ingestion(self, file_path: str) -> None:
        """Requirement: Log the whole process from start to finish."""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        success_count = 0
        total_count = 0

        try:
            logger.info(f"PROCESS START: Ingesting {file_path}")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Source file not found: {file_path}")

            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    total_count += 1

                    try:
                        record = json.loads(line)
                        if self.validate_record(record):
                            cursor.execute(
                                f"INSERT INTO {self.table_name} (RAW_CONTENT) VALUES (?)",
                                (json.dumps(record),)
                            )
                            success_count += 1
                    except json.JSONDecodeError:
                        logger.warning(f"Line {total_count}: Invalid JSON format skipped.")

            self.conn.commit()
            logger.info(f"PROCESS SUCCESS: {success_count}/{total_count} records ingested.")

        except Exception as e:
            logger.error(f"PROCESS FAILED: {str(e)}")
            if self.conn: self.conn.rollback()
        finally:
            if self.conn: self.conn.close()
            logger.info("PROCESS FINALIZED: Resources released.")