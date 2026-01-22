import os
import logging
import sys

def setup_logging():
    # Use lowercase name as requested in the new implementation
    logger = logging.getLogger("autoeda")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Detect if running in AWS Lambda
    IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None

    if IS_LAMBDA:
        # 1. DISABLE FILE LOGGING: Do not use RotatingFileHandler here
        # 2. REDIRECT TO STDOUT: Lambda captures anything sent to the console
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s | %(name)s | %(message)s')
    else:
        # Standard local file logging
        from logging.handlers import RotatingFileHandler
        os.makedirs("logs", exist_ok=True)
        handler = RotatingFileHandler(
            "logs/autoeda.log", 
            maxBytes=10*1024*1024, 
            backupCount=5
        )
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Create a default logger instance for imports
logger = setup_logging()