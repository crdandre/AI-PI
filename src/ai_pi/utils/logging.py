from pathlib import Path
import logging
from datetime import datetime

def setup_logging(log_dir: Path, timestamp: str, logger_name: str = None) -> logging.Logger:
    """
    Centralized logging configuration for the AI-PI system.
    
    Args:
        log_dir: Directory where log files will be stored
        timestamp: Timestamp string for log file naming
        logger_name: Optional name for the logger (defaults to timestamp-based name)
    
    Returns:
        Configured logger instance
    """
    if logger_name is None:
        logger_name = f"ai_pi_{timestamp}"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log file
    log_file = log_dir / f"{logger_name}_{timestamp}.log"
    
    # File handler with detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Console handler with simplified output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 