"""
Universal AI Assistant - Few-Shot Multi-Task Learning Framework

This package provides a comprehensive framework for building universal AI assistants
that can rapidly adapt to new tasks using few-shot learning techniques.
"""

__version__ = "0.1.0"
__author__ = "Vishnu Project Team"

# Core imports
from .models.universal_assistant import UniversalAssistant, create_universal_assistant
from .core.meta_learner import create_meta_learner, MetaLearningAlgorithm
from .tasks.base import TaskType, TaskDomain, Modality, BaseTask

__all__ = [
    "UniversalAssistant",
    "create_universal_assistant",
    "create_meta_learner", 
    "MetaLearningAlgorithm",
    "TaskType",
    "TaskDomain", 
    "Modality",
    "BaseTask"
]
