import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(level=None):
    """
    Setup centralized logging configuration.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger("AutoEDA")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []

    # File Handler with rotation
    file_handler = RotatingFileHandler(
        'logs/autoeda.log', 
        maxBytes=10*1024*1024, # 10MB
        backupCount=5
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# Create a default logger instance
logger = setup_logging()
