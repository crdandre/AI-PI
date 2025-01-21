from pathlib import Path
import logging
import functools
import time

def get_logger(name: str, processor=None) -> logging.Logger:
    """Helper function to get logger and format LLM info if available"""
    logger = logging.getLogger(name)
    llm_info = f" using {processor.step.lm_name}" if processor and hasattr(processor.step, 'lm_name') else ""
    return logger, llm_info

def log_step(logger_name: str = None):
    """Decorator to log the start and end of processing steps"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            processor = args[0]
            logger, llm_info = get_logger(logger_name or processor.__class__.__name__, processor)
            
            logger.info(f"Starting step: {func.__name__}{llm_info}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed step: {func.__name__}{llm_info} (duration: {time.time() - start_time:.2f}s)")
                return result
            except Exception as e:
                logger.error(f"Failed step: {func.__name__}{llm_info} - {str(e)}")
                raise
        return wrapper
    return decorator

def setup_logging(log_dir: Path, timestamp: str, logger_name: str = None) -> logging.Logger:
    """Centralized logging configuration"""
    logger_name = logger_name or f"ai_pi_{timestamp}"
    logger = logging.getLogger(logger_name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    log_file = log_dir / f"{logger_name}_{timestamp}.log"
    
    handlers = [
        (logging.FileHandler(log_file), logging.DEBUG, '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        (logging.StreamHandler(), logging.INFO, '%(levelname)s: %(message)s')
    ]
    
    for handler, level, format in handlers:
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format))
        logger.addHandler(handler)
    
    return logger

def debug_output(logger_name: str = None):
    """Decorator for debug output logging"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if self.step.verbose:
                logger = logging.getLogger(logger_name or self.__class__.__name__)
                logger.debug(f"{logger.name}: {result}")
            return result
        return wrapper
    return decorator 