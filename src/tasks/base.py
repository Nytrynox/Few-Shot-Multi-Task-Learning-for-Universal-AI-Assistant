"""
Base classes and enums for task definitions.

This module defines the core task types and interfaces used throughout
the universal assistant system.
"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn


class TaskType(Enum):
    """Enumeration of supported task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    SEQUENCE_LABELING = "sequence_labeling"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_COMPLETION = "code_completion"
    CODE_GENERATION = "code_generation"
    MATH_SOLVING = "math_solving"
    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "image_classification"
    VISUAL_QA = "visual_qa"


class TaskDomain(Enum):
    """Enumeration of task domains."""
    NLP = "nlp"
    VISION = "vision"
    CODE = "code"
    MATH = "math"
    MULTIMODAL = "multimodal"
    AUDIO = "audio"


class Modality(Enum):
    """Enumeration of input modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"
    MULTIMODAL = "multimodal"


class BaseTask(ABC):
    """
    Abstract base class for all tasks in the universal assistant system.
    """
    
    def __init__(
        self,
        name: str,
        task_type: TaskType,
        domain: TaskDomain,
        modality: Modality,
        description: str = ""
    ):
        self.name = name
        self.task_type = task_type
        self.domain = domain
        self.modality = modality
        self.description = description
    
    @abstractmethod
    def preprocess_input(self, raw_input: Any) -> Dict[str, torch.Tensor]:
        """
        Preprocess raw input for the model.
        
        Args:
            raw_input: Raw input data
            
        Returns:
            Preprocessed input tensors
        """
        pass
    
    @abstractmethod
    def postprocess_output(self, model_output: torch.Tensor) -> Any:
        """
        Postprocess model output to final result.
        
        Args:
            model_output: Raw model output
            
        Returns:
            Final processed result
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute task-specific loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss tensor
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        predictions: List[Any],
        targets: List[Any]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against targets.
        
        Args:
            predictions: List of predictions
            targets: List of ground truth targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def create_few_shot_episode(
        self,
        support_examples: List[Dict[str, Any]],
        query_examples: List[Dict[str, Any]],
        n_way: int,
        k_shot: int
    ) -> Dict[str, torch.Tensor]:
        """
        Create a few-shot learning episode.
        
        Args:
            support_examples: Support set examples
            query_examples: Query set examples
            n_way: Number of classes
            k_shot: Number of examples per class
            
        Returns:
            Episode data as tensors
        """
        # Default implementation - can be overridden by specific tasks
        support_inputs = []
        support_targets = []
        query_inputs = []
        query_targets = []
        
        for example in support_examples:
            inputs = self.preprocess_input(example['input'])
            support_inputs.append(inputs)
            support_targets.append(example['target'])
        
        for example in query_examples:
            inputs = self.preprocess_input(example['input'])
            query_inputs.append(inputs)
            query_targets.append(example['target'])
        
        return {
            'support_x': torch.stack([inp['input_ids'] for inp in support_inputs]) if support_inputs else torch.empty(0),
            'support_y': torch.tensor(support_targets),
            'query_x': torch.stack([inp['input_ids'] for inp in query_inputs]) if query_inputs else torch.empty(0),
            'query_y': torch.tensor(query_targets),
            'n_way': n_way,
            'k_shot': k_shot
        }


class TaskFactory:
    """
    Factory class for creating task instances.
    """
    
    _task_registry = {}
    
    @classmethod
    def register_task(cls, task_class: type):
        """Register a new task class."""
        cls._task_registry[task_class.__name__.lower()] = task_class
        return task_class
    
    @classmethod
    def create_task(cls, task_name: str, **kwargs) -> BaseTask:
        """Create a task instance by name."""
        task_name = task_name.lower()
        if task_name not in cls._task_registry:
            raise ValueError(f"Unknown task: {task_name}")
        
        task_class = cls._task_registry[task_name]
        return task_class(**kwargs)
    
    @classmethod
    def list_available_tasks(cls) -> List[str]:
        """List all available task names."""
        return list(cls._task_registry.keys())


class TaskMetrics:
    """
    Collection of common evaluation metrics for different task types.
    """
    
    @staticmethod
    def accuracy(predictions: List[int], targets: List[int]) -> float:
        """Compute accuracy for classification tasks."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        correct = sum(p == t for p, t in zip(predictions, targets))
        return correct / len(predictions) if len(predictions) > 0 else 0.0
    
    @staticmethod
    def f1_score(predictions: List[int], targets: List[int], average: str = 'macro') -> float:
        """Compute F1 score for classification tasks."""
        # Simplified F1 computation - in practice, use sklearn
        from collections import defaultdict
        
        # Count true positives, false positives, false negatives per class
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        
        for p, t in zip(predictions, targets):
            if p == t:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1
        
        # Compute F1 for each class
        f1_scores = []
        for class_id in set(targets):
            precision = tp[class_id] / (tp[class_id] + fp[class_id]) if (tp[class_id] + fp[class_id]) > 0 else 0
            recall = tp[class_id] / (tp[class_id] + fn[class_id]) if (tp[class_id] + fn[class_id]) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    @staticmethod
    def mse(predictions: List[float], targets: List[float]) -> float:
        """Compute mean squared error for regression tasks."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    
    @staticmethod
    def mae(predictions: List[float], targets: List[float]) -> float:
        """Compute mean absolute error for regression tasks."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)
    
    @staticmethod
    def bleu_score(predictions: List[str], targets: List[str]) -> float:
        """Simplified BLEU score for generation tasks."""
        # This is a very simplified version - use nltk.translate.bleu_score in practice
        total_score = 0.0
        
        for pred, target in zip(predictions, targets):
            pred_words = pred.lower().split()
            target_words = target.lower().split()
            
            if not pred_words or not target_words:
                continue
            
            # Compute 1-gram precision
            matches = sum(1 for word in pred_words if word in target_words)
            precision = matches / len(pred_words) if pred_words else 0
            
            # Simple length penalty
            length_penalty = min(1.0, len(pred_words) / len(target_words)) if target_words else 0
            
            total_score += precision * length_penalty
        
        return total_score / len(predictions) if predictions else 0.0


class TaskConfig:
    """
    Configuration class for tasks.
    """
    
    def __init__(
        self,
        name: str,
        task_type: TaskType,
        domain: TaskDomain,
        modality: Modality,
        num_classes: Optional[int] = None,
        max_length: Optional[int] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        evaluation_metrics: Optional[List[str]] = None
    ):
        self.name = name
        self.task_type = task_type
        self.domain = domain
        self.modality = modality
        self.num_classes = num_classes
        self.max_length = max_length
        self.preprocessing_config = preprocessing_config or {}
        self.evaluation_metrics = evaluation_metrics or self._get_default_metrics()
    
    def _get_default_metrics(self) -> List[str]:
        """Get default evaluation metrics for the task type."""
        if self.task_type in [TaskType.CLASSIFICATION, TaskType.IMAGE_CLASSIFICATION]:
            return ['accuracy', 'f1_score']
        elif self.task_type == TaskType.REGRESSION:
            return ['mse', 'mae']
        elif self.task_type in [TaskType.GENERATION, TaskType.SUMMARIZATION, TaskType.TRANSLATION]:
            return ['bleu_score']
        else:
            return ['accuracy']
