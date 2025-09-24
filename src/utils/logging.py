"""
Logging utilities for the Universal AI Assistant project.

This module provides centralized logging configuration and utilities.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set logging level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def setup_file_logging(
    log_file: str,
    level: str = "INFO",
    logger_name: Optional[str] = None
):
    """
    Set up file logging for a specific logger or root logger.
    
    Args:
        log_file: Path to log file
        level: Logging level
        logger_name: Specific logger name (None for root logger)
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get logger
    logger = logging.getLogger(logger_name)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)


def configure_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_file: Optional[str] = None
):
    """
    Configure global logging settings.
    
    Args:
        console_level: Console logging level
        file_level: File logging level
        log_file: Optional log file path
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        setup_file_logging(log_file, file_level)
