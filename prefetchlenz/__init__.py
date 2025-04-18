# prefetchlenz/__init__.py

__version__ = "0.1.0"

# --- Logging Setup ---
import logging

logger = logging.getLogger("prefetchlenz")
logger.setLevel(logging.INFO)  # Change to DEBUG to see more info

# Avoid adding multiple handlers during reloads (like in Jupyter)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("PrefetchLenz Library Initialized")
