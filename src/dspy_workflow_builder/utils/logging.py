from pathlib import Path
import logging
from datetime import datetime
import functools
import time

def log_step(logger_name: str = None):
    """
    Decorator to log the start and end of processing steps, including LLM info if available.
    
    Args:
        logger_name: Optional name for the logger. If None, uses the class name
    
    Returns:
        Decorator function that logs step execution
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger name from parameter or class name
            _logger_name = logger_name or args[0].__class__.__name__
            logger = logging.getLogger(_logger_name)
            
            # Get LLM info from step configuration if it's an LMStep
            processor = args[0]
            llm_info = ""
            if hasattr(processor.step, 'lm_name'):
                llm_info = f" using {processor.step.lm_name}"
            
            # Log start
            logger.info(f"Starting step: {func.__name__}{llm_info}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                # Log successful completion
                duration = time.time() - start_time
                logger.info(f"Completed step: {func.__name__}{llm_info} (duration: {duration:.2f}s)")
                return result
            except Exception as e:
                # Log failure
                logger.error(f"Failed step: {func.__name__}{llm_info} - {str(e)}")
                raise
                
        return wrapper
    return decorator

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