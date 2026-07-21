"""
Application logging utilities using Loguru.
"""

import sys
import warnings
from loguru import logger

# Suppress noisy SciPy / scikit-learn UserWarnings about local NumPy version mismatches
warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy version.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*sklearn.*")

# Suppress noisy third-party logging to avoid "red words" in stderr
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

# Remove default handler and configure clean colored formatting with Loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[module]}</cyan>: <level>{message}</level>",
    level="INFO",
)


def get_logger(name: str):
    """Get a Loguru logger instance bound with module name."""
    return logger.bind(module=name)
