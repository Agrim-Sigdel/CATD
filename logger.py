"""Shared logger — writes to file and optionally to a Qt signal for the viewer."""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_FILE = os.getenv("LOG_FILE", "anpr.log")

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Rotating file — max 5 MB, keep 3 backups
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
