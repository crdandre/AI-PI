from pathlib import Path
import logging
import functools
import time
from contextlib import contextmanager
from enum import Enum

# Track pipeline nesting globally (reset on each pipeline execution)
class LogContext:
    _pipeline_stack = []
    
    @classmethod
    def push_pipeline(cls, name):
        cls._pipeline_stack.append(name)
    
    @classmethod
    def pop_pipeline(cls):
        if cls._pipeline_stack:
            cls._pipeline_stack.pop()
    
    @classmethod
    def get_indent(cls):
        return "  " * len(cls._pipeline_stack)
    
    @classmethod
    def reset(cls):
        cls._pipeline_stack = []

def get_logger(name: str, processor=None) -> tuple[logging.Logger, str]:
    """Helper function to get logger and format LLM info if available"""
    logger = logging.getLogger(name)
    llm_info = f" using {processor.step.lm_name}" if processor and hasattr(processor.step, 'lm_name') else ""
    return logger, llm_info

@contextmanager
def pipeline_context(name):
    """Context manager for pipeline execution"""
    try:
        LogContext.push_pipeline(name)
        yield
    finally:
        LogContext.pop_pipeline()

def log_step(logger_name: str = None):
    """Decorator to log the start and end of processing steps"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            processor = args[0]
            # Get processor name if no logger_name provided
            actual_logger_name = logger_name or processor.__class__.__name__
            logger = logging.getLogger(actual_logger_name)
            
            # Get current pipeline context indent
            indent = LogContext.get_indent()
            
            # Get step name, handling both Enum and string types
            step_type = processor.step.step_type
            if isinstance(step_type, Enum):
                step_name = step_type.value
            else:
                step_name = str(step_type)
            
            logger.info(f"{indent}┌─ Starting step: {step_name}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{indent}└─ Completed step: {step_name} ({duration:.2f}s)")
                return result
            except Exception as e:
                logger.error(f"{indent}└─ Failed step: {step_name} - {str(e)}")
                raise
        return wrapper
    return decorator

def setup_logging(log_dir: Path, timestamp: str, logger_name: str = None) -> logging.Logger:
    """Centralized logging configuration"""
    # Reset the LogContext for new pipeline execution
    LogContext.reset()
    
    # Configure root logger to catch all logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Setup handlers for root logger
    handlers = [
        (logging.FileHandler(log_dir / f"ai_pi_{timestamp}.log"), logging.DEBUG, 
         '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        (logging.StreamHandler(), logging.INFO, '%(message)s')  # Simplified console format
    ]
    
    for handler, level, format in handlers:
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format))
        root_logger.addHandler(handler)
    
    # Get specific logger if requested
    if logger_name:
        return logging.getLogger(logger_name)
    return root_logger

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