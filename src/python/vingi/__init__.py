"""
Vingi Personal Cognitive Load Optimization Framework

A privacy-first AI assistant for automating personal affairs and reducing cognitive load.
"""

__version__ = "1.0.0-beta"
__author__ = "Vingi Team"
__email__ = "team@vingi.dev"

# Core components
from .core.context_graph import ContextGraphManager
from .core.pattern_recognition import TemporalPatternRecognizer
from .core.relevance_scoring import RelevanceScorer

# NLP components
from .nlp.email_classifier import EmailClassifier
from .nlp.text_summarizer import TextSummarizer
from .nlp.intent_recognition import IntentRecognizer

# Research components
from .research.search_agent import ContextualResearchAgent
from .research.citation_parser import CitationParser
from .research.knowledge_synthesis import KnowledgeSynthesizer

# Automation components
from .automation.file_manager import SemanticFileOrganizer
from .automation.email_processor import EmailProcessor
from .automation.calendar_optimizer import CalendarOptimizer

# Utilities
from .utils.config_loader import ConfigLoader
from .utils.logging import get_logger
from .utils.encryption import DataEncryptor

# Bridge interface
from .bridge.swift_interface import SwiftBridgeServer

__all__ = [
    # Core
    "ContextGraphManager",
    "TemporalPatternRecognizer", 
    "RelevanceScorer",
    
    # NLP
    "EmailClassifier",
    "TextSummarizer",
    "IntentRecognizer",
    
    # Research
    "ContextualResearchAgent",
    "CitationParser",
    "KnowledgeSynthesizer",
    
    # Automation
    "SemanticFileOrganizer",
    "EmailProcessor",
    "CalendarOptimizer",
    
    # Utilities
    "ConfigLoader",
    "get_logger",
    "DataEncryptor",
    
    # Bridge
    "SwiftBridgeServer",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]

# Package-level configuration
import logging
import os
from pathlib import Path

# Set up package-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Default data directory
DEFAULT_DATA_DIR = Path.home() / "Library" / "Application Support" / "Vingi"
DATA_DIR = Path(os.getenv("VINGI_DATA_PATH", DEFAULT_DATA_DIR))

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(DATA_DIR / "models").mkdir(exist_ok=True)
(DATA_DIR / "cache").mkdir(exist_ok=True)
(DATA_DIR / "logs").mkdir(exist_ok=True)
(DATA_DIR / "context").mkdir(exist_ok=True)

def get_data_dir() -> Path:
    """Get the current data directory path."""
    return DATA_DIR

def get_config_dir() -> Path:
    """Get the configuration directory path."""
    config_dir = Path.home() / ".config" / "vingi"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
