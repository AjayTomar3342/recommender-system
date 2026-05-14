import subprocess
import os
import shutil
from src.utils.logger import setup_custom_logger

logger = setup_custom_logger("TransformationModule")

class DBTRunner:
    def __init__(self, dbt_dir: str):
        # This points to /Users/ajaytomar/PycharmProjects/recommender-system/dbt_transform
        self.dbt_dir = os.path.abspath(dbt_dir)

    def run_command(self, command: list) -> bool:
        """Helper to run dbt commands and log output."""
        # Find the absolute path to dbt in the .venv
        dbt_path = shutil.which("dbt")
        if not dbt_path:
            dbt_path = "/Users/ajaytomar/PycharmProjects/recommender-system/.venv/bin/dbt"

        cmd = [dbt_path] + command + ["--profiles-dir", "."]
        logger.info(f"Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.dbt_dir,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"DBT {command[0]} Successful")
                print(result.stdout)
                return True
            else:
                logger.error(f"DBT {command[0]} Failed!")
                print(result.stdout)
                print(result.stderr)
                return False
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False

    def run_pipeline(self):
        """Executes the full dbt lifecycle (deps -> run -> test)."""
        logger.info("Starting DBT Transformation Pipeline...")

        # 1. Install packages (dbt_utils)
        if not self.run_command(["deps"]):
            return

        # 2. Run the models (stg_amazon_reviews -> fact_reviews)
        if not self.run_command(["run"]):
            return

        # 3. Run the tests (checks for nulls and rating ranges)
        if not self.run_command(["test"]):
            return

        logger.info("✅ ALL STEPS COMPLETE: Pipeline finished successfully.")