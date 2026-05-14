import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# =============================================================================
# 1. MAC OS GPU FIX
# Prevents the crash: "MPSGraphObject initialize may have been in progress..."
# =============================================================================
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# =============================================================================
# 2. PATH CONFIGURATION
# Adds your project root so Airflow can find 'main.py' and your modules
# =============================================================================
PROJECT_ROOT = '/Users/ajaytomar/PycharmProjects/recommender-system'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# =============================================================================
# 3. IMPORT LOGIC
# Now that the path is set, we can import the 'main' function
# =============================================================================
try:
    from main import main
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback to prevent DAG from breaking entirely during parsing
    def main():
        raise ImportError(f"Could not find main function. Check path: {PROJECT_ROOT}")

# =============================================================================
# 4. DAG DEFINITION
# =============================================================================
with DAG(
    dag_id="execute_my_main_project",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Changed to None for manual triggering (easier for testing)
    catchup=False
) as dag:

    run_task = PythonOperator(
        task_id="run_main_script",
        python_callable=main  # Correctly matches the 'from main import main'
    )