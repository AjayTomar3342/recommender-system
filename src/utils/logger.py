import logging
import sys
import os
import uuid
import subprocess
from datetime import datetime
from typing import Optional


def get_git_revision() -> str:
    """Requirement G: Capture version control details."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "No-Git-Commit"


def setup_custom_logger(module_name: str) -> logging.Logger:
    """
    I/P: module_name (str)
    O/P: logger (logging.Logger)
    Behaviour: Creates/Appends to a .txt file in /logs. Distinguishes sessions via UUID.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Session Metadata
    session_id = str(uuid.uuid4())[:8]
    git_hash = get_git_revision()

    # Format: SessionID | Time | Module | Level | Message
    formatter = logging.Formatter(f'[{session_id}] [%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')

    log_filename = f"{log_dir}/process_log.txt"
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(formatter)

    if not logger.handlers:
        # Add headers to the file specifically for new session start
        with open(log_filename, 'a') as f:
            f.write(f"\n{'=' * 80}\nSESSION START: {session_id} | GIT: {git_hash} | {datetime.now()}\n{'=' * 80}\n")

        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger