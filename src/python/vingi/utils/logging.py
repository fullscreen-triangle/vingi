"""
Logging utilities for Vingi

Provides structured logging with privacy-aware log filtering.
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import re

from .. import get_data_dir


class PrivacyLogFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""
    
    SENSITIVE_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
        r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN patterns
        r'password["\']?\s*[:=]\s*["\']?[^"\'>\s]+',  # Passwords
        r'token["\']?\s*[:=]\s*["\']?[^"\'>\s]+',  # Tokens
        r'key["\']?\s*[:=]\s*["\']?[^"\'>\s]+',  # API keys
    ]
    
    def __init__(self):
        super().__init__()
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SENSITIVE_PATTERNS]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive information from log records."""
        if hasattr(record, 'msg'):
            message = str(record.msg)
            for pattern in self.patterns:
                message = pattern.sub('[REDACTED]', message)
            record.msg = message
        
        # Also filter args if present
        if hasattr(record, 'args') and record.args:
            filtered_args = []
            for arg in record.args:
                arg_str = str(arg)
                for pattern in self.patterns:
                    arg_str = pattern.sub('[REDACTED]', arg_str)
                filtered_args.append(arg_str)
            record.args = tuple(filtered_args)
        
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class VingiLogger:
    """Main logger class for Vingi."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        log_file: Optional[Path] = None,
        enable_console: bool = True,
        enable_json: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_privacy_filter: bool = True
    ) -> None:
        """Configure logging for the entire Vingi package."""
        
        if cls._configured:
            return
        
        # Set root logger level
        root_logger = logging.getLogger('vingi')
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Privacy filter
        privacy_filter = PrivacyLogFilter() if enable_privacy_filter else None
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if enable_json:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            
            if privacy_filter:
                console_handler.addFilter(privacy_filter)
            
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file is None:
            log_file = get_data_dir() / "logs" / "vingi.log"
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        
        if enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
            )
        
        if privacy_filter:
            file_handler.addFilter(privacy_filter)
        
        root_logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if not cls._configured:
            cls.configure()
        
        if name not in cls._loggers:
            logger = logging.getLogger(f'vingi.{name}')
            cls._loggers[name] = logger
        
        return cls._loggers[name]


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return VingiLogger.get_logger(name)


def configure_logging(
    level: str = None,
    log_file: Optional[str] = None,
    enable_console: bool = None,
    enable_json: bool = None,
    enable_privacy_filter: bool = None
) -> None:
    """Configure logging with environment variable support."""
    
    # Get configuration from environment variables
    env_level = os.getenv('VINGI_LOG_LEVEL', 'INFO')
    env_log_file = os.getenv('VINGI_LOG_FILE')
    env_enable_console = os.getenv('VINGI_LOG_CONSOLE', 'true').lower() == 'true'
    env_enable_json = os.getenv('VINGI_LOG_JSON', 'false').lower() == 'true'
    env_enable_privacy = os.getenv('VINGI_LOG_PRIVACY_FILTER', 'true').lower() == 'true'
    
    # Use provided values or fall back to environment/defaults
    final_level = level or env_level
    final_log_file = Path(log_file) if log_file else (Path(env_log_file) if env_log_file else None)
    final_enable_console = enable_console if enable_console is not None else env_enable_console
    final_enable_json = enable_json if enable_json is not None else env_enable_json
    final_enable_privacy = enable_privacy_filter if enable_privacy_filter is not None else env_enable_privacy
    
    VingiLogger.configure(
        level=final_level,
        log_file=final_log_file,
        enable_console=final_enable_console,
        enable_json=final_enable_json,
        enable_privacy_filter=final_enable_privacy
    )


class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls with arguments and return values."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Log function entry
            logger.debug(
                f"Entering {func_name}",
                extra={
                    'function': func_name,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'event': 'function_entry'
                }
            )
            
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.debug(
                    f"Completed {func_name}",
                    extra={
                        'function': func_name,
                        'duration_seconds': duration,
                        'event': 'function_success'
                    }
                )
                
                return result
                
            except Exception as e:
                # Log exception
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.error(
                    f"Exception in {func_name}: {str(e)}",
                    extra={
                        'function': func_name,
                        'duration_seconds': duration,
                        'exception_type': type(e).__name__,
                        'event': 'function_error'
                    },
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


def setup_performance_logging(logger: logging.Logger, operation: str):
    """Context manager for performance logging."""
    
    class PerformanceLogger:
        def __init__(self, logger: logging.Logger, operation: str):
            self.logger = logger
            self.operation = operation
            self.start_time = None
        
        def __enter__(self):
            self.start_time = datetime.now()
            self.logger.info(
                f"Starting {self.operation}",
                extra={'operation': self.operation, 'event': 'operation_start'}
            )
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            if exc_type is None:
                self.logger.info(
                    f"Completed {self.operation}",
                    extra={
                        'operation': self.operation,
                        'duration_seconds': duration,
                        'event': 'operation_success'
                    }
                )
            else:
                self.logger.error(
                    f"Failed {self.operation}: {str(exc_val)}",
                    extra={
                        'operation': self.operation,
                        'duration_seconds': duration,
                        'exception_type': exc_type.__name__,
                        'event': 'operation_error'
                    }
                )
    
    return PerformanceLogger(logger, operation)


# Initialize logging when module is imported
configure_logging() 