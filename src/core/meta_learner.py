"""
Core meta-learning interface and factory for different algorithms.

This module provides a unified interface for various meta-learning algorithms
and includes factory methods for creating meta-learners.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
from enum import Enum

from .maml import MAMLLearner, MAMLPlusPlus
from .prototypical import PrototypicalNetworks, RelationNetworks
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetaLearningAlgorithm(Enum):
    """Enumeration of supported meta-learning algorithms."""
    MAML = "maml"
    MAML_PLUS_PLUS = "maml++"
    PROTOTYPICAL = "prototypical"
    RELATION = "relation"
    REPTILE = "reptile"
    ANIL = "anil"


class BaseMetaLearner(ABC):
    """Abstract base class for meta-learning algorithms."""
    
    @abstractmethod
    def adapt(
        self, 
        support_x: torch.Tensor, 
        support_y: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Adapt to a new task using support examples."""
        pass
    
    @abstractmethod
    def predict(
        self,
        query_x: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Make predictions on query examples after adaptation."""
        pass
    
    @abstractmethod
    def meta_update(
        self,
        task_batch: List[Dict[str, torch.Tensor]],
        **kwargs
    ) -> Dict[str, float]:
        """Perform meta-learning update across multiple tasks."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        pass


class MetaLearnerWrapper(BaseMetaLearner):
    """
    Wrapper class that provides a unified interface for different meta-learning algorithms.
    """
    
    def __init__(self, meta_learner: Union[MAMLLearner, PrototypicalNetworks]):
        self.meta_learner = meta_learner
        self.algorithm_type = self._detect_algorithm_type()
        
    def _detect_algorithm_type(self) -> str:
        """Detect the type of meta-learning algorithm."""
        if isinstance(self.meta_learner, MAMLLearner):
            return "gradient_based"
        elif isinstance(self.meta_learner, PrototypicalNetworks):
            return "metric_based"
        elif isinstance(self.meta_learner, RelationNetworks):
            return "metric_based"
        else:
            return "unknown"
    
    def adapt(
        self, 
        support_x: torch.Tensor, 
        support_y: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Unified adaptation interface."""
        if self.algorithm_type == "gradient_based":
            return self.meta_learner.adapt(support_x, support_y, **kwargs)
        elif self.algorithm_type == "metric_based":
            # For metric-based methods, adaptation is implicit in the forward pass
            return {"status": "adapted", "algorithm": "metric_based"}
        else:
            raise NotImplementedError(f"Adaptation not implemented for {type(self.meta_learner)}")
    
    def predict(
        self,
        query_x: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Unified prediction interface."""
        if hasattr(self.meta_learner, 'predict'):
            return self.meta_learner.predict(support_x, support_y, query_x, **kwargs)
        else:
            # For gradient-based methods, need to adapt first then predict
            adapted_result = self.adapt(support_x, support_y, return_adapted_model=True)
            adapted_model = adapted_result.get('adapted_model')
            if adapted_model:
                with torch.no_grad():
                    return adapted_model(query_x)
            else:
                raise RuntimeError("Could not obtain adapted model for prediction")
    
    def meta_update(
        self,
        task_batch: List[Dict[str, torch.Tensor]],
        **kwargs
    ) -> Dict[str, float]:
        """Unified meta-update interface."""
        return self.meta_learner.meta_update(task_batch, **kwargs)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        self.meta_learner.save_checkpoint(path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self.meta_learner.load_checkpoint(path)


def create_meta_learner(
    algorithm: Union[str, MetaLearningAlgorithm],
    model: nn.Module,
    **kwargs
) -> BaseMetaLearner:
    """
    Factory function to create meta-learning algorithms.
    
    Args:
        algorithm: Meta-learning algorithm to use
        model: Base neural network model
        **kwargs: Algorithm-specific arguments
        
    Returns:
        Configured meta-learner instance
    """
    if isinstance(algorithm, str):
        algorithm = MetaLearningAlgorithm(algorithm.lower())
    
    if algorithm == MetaLearningAlgorithm.MAML:
        meta_learner = MAMLLearner(
            model=model,
            inner_lr=kwargs.get('inner_lr', 0.01),
            outer_lr=kwargs.get('outer_lr', 0.001),
            inner_steps=kwargs.get('inner_steps', 5),
            first_order=kwargs.get('first_order', False)
        )
        
    elif algorithm == MetaLearningAlgorithm.MAML_PLUS_PLUS:
        meta_learner = MAMLPlusPlus(
            model=model,
            inner_lr=kwargs.get('inner_lr', 0.01),
            outer_lr=kwargs.get('outer_lr', 0.001),
            inner_steps=kwargs.get('inner_steps', 5),
            multi_step_loss=kwargs.get('multi_step_loss', True),
            per_param_lr=kwargs.get('per_param_lr', True)
        )
        
    elif algorithm == MetaLearningAlgorithm.PROTOTYPICAL:
        meta_learner = PrototypicalNetworks(
            backbone=model,
            distance_metric=kwargs.get('distance_metric', 'euclidean'),
            temperature=kwargs.get('temperature', 1.0)
        )
        
    elif algorithm == MetaLearningAlgorithm.RELATION:
        relation_module = kwargs.get('relation_module')
        if relation_module is None:
            # Create default relation module
            embed_dim = kwargs.get('embed_dim', 512)
            relation_module = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            )
        
        meta_learner = RelationNetworks(
            backbone=model,
            relation_module=relation_module,
            temperature=kwargs.get('temperature', 1.0)
        )
        
    else:
        raise ValueError(f"Unsupported meta-learning algorithm: {algorithm}")
    
    logger.info(f"Created {algorithm.value} meta-learner with model {type(model).__name__}")
    return MetaLearnerWrapper(meta_learner)


class MultiTaskMetaLearner:
    """
    Meta-learner that handles multiple task types simultaneously.
    
    This class coordinates different meta-learners for different task domains
    and provides task-specific routing.
    """
    
    def __init__(self, task_meta_learners: Dict[str, BaseMetaLearner]):
        """
        Args:
            task_meta_learners: Dictionary mapping task types to meta-learners
        """
        self.task_meta_learners = task_meta_learners
        self.supported_tasks = list(task_meta_learners.keys())
        
    def adapt(
        self,
        task_type: str,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Adapt to a task of specified type."""
        if task_type not in self.supported_tasks:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        meta_learner = self.task_meta_learners[task_type]
        return meta_learner.adapt(support_x, support_y, **kwargs)
    
    def predict(
        self,
        task_type: str,
        query_x: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Make predictions for a task of specified type."""
        if task_type not in self.supported_tasks:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        meta_learner = self.task_meta_learners[task_type]
        return meta_learner.predict(query_x, support_x, support_y, **kwargs)
    
    def meta_update(
        self,
        multi_task_batch: Dict[str, List[Dict[str, torch.Tensor]]],
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Perform meta-updates across multiple task types."""
        results = {}
        
        for task_type, task_batch in multi_task_batch.items():
            if task_type in self.supported_tasks:
                meta_learner = self.task_meta_learners[task_type]
                results[task_type] = meta_learner.meta_update(task_batch, **kwargs)
            else:
                logger.warning(f"Skipping unsupported task type: {task_type}")
        
        return results
    
    def save_checkpoints(self, base_path: str):
        """Save checkpoints for all meta-learners."""
        for task_type, meta_learner in self.task_meta_learners.items():
            checkpoint_path = f"{base_path}_{task_type}.pt"
            meta_learner.save_checkpoint(checkpoint_path)
    
    def load_checkpoints(self, base_path: str):
        """Load checkpoints for all meta-learners."""
        for task_type, meta_learner in self.task_meta_learners.items():
            checkpoint_path = f"{base_path}_{task_type}.pt"
            try:
                meta_learner.load_checkpoint(checkpoint_path)
            except FileNotFoundError:
                logger.warning(f"Checkpoint not found for task type: {task_type}")


def create_multi_task_meta_learner(
    task_configs: Dict[str, Dict],
    shared_backbone: Optional[nn.Module] = None
) -> MultiTaskMetaLearner:
    """
    Create a multi-task meta-learner with task-specific configurations.
    
    Args:
        task_configs: Dictionary mapping task types to their configurations
        shared_backbone: Optional shared backbone model across tasks
        
    Returns:
        Configured multi-task meta-learner
    """
    task_meta_learners = {}
    
    for task_type, config in task_configs.items():
        # Use shared backbone if provided, otherwise create task-specific model
        model = shared_backbone if shared_backbone is not None else config['model']
        
        # Create meta-learner for this task type
        meta_learner = create_meta_learner(
            algorithm=config['algorithm'],
            model=model,
            **config.get('kwargs', {})
        )
        
        task_meta_learners[task_type] = meta_learner
    
    logger.info(f"Created multi-task meta-learner for tasks: {list(task_configs.keys())}")
    return MultiTaskMetaLearner(task_meta_learners)
