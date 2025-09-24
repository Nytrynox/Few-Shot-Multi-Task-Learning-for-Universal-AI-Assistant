"""
Universal AI Assistant Model Architecture.

This module implements the main Universal AI Assistant that combines
few-shot learning capabilities with multi-task learning across different domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import AutoModel, AutoTokenizer
import numpy as np

from ..core.meta_learner import create_meta_learner, MultiTaskMetaLearner
from ..tasks.base import TaskType, BaseTask
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MultiModalBackbone(nn.Module):
    """
    Multi-modal backbone network that can handle text, images, and code inputs.
    """
    
    def __init__(
        self,
        text_model_name: str = "microsoft/DialoGPT-medium",
        vision_model_name: str = "google/vit-base-patch16-224",
        hidden_dim: int = 768,
        output_dim: int = 512
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Text encoder (transformer-based)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, output_dim)
        
        # Vision encoder
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        self.vision_projection = nn.Linear(self.vision_encoder.config.hidden_size, output_dim)
        
        # Code encoder (can reuse text encoder with different projection)
        self.code_projection = nn.Linear(self.text_encoder.config.hidden_size, output_dim)
        
        # Cross-modal attention for fusion
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final projection layers
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode text inputs."""
        outputs = self.text_encoder(**text_inputs)
        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return self.text_projection(embeddings)
    
    def encode_vision(self, image_inputs: torch.Tensor) -> torch.Tensor:
        """Encode vision inputs."""
        outputs = self.vision_encoder(pixel_values=image_inputs)
        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        return self.vision_projection(embeddings)
    
    def encode_code(self, code_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode code inputs (using text encoder with different projection)."""
        outputs = self.text_encoder(**code_inputs)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return self.code_projection(embeddings)
    
    def forward(
        self, 
        inputs: Dict[str, Any],
        modality: str = "text"
    ) -> torch.Tensor:
        """
        Forward pass for different modalities.
        
        Args:
            inputs: Input data (format depends on modality)
            modality: Input modality ("text", "vision", "code")
            
        Returns:
            Encoded representations
        """
        if modality == "text":
            embeddings = self.encode_text(inputs)
        elif modality == "vision":
            embeddings = self.encode_vision(inputs)
        elif modality == "code":
            embeddings = self.encode_code(inputs)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Apply final projection
        return self.output_projection(embeddings)


class TaskSpecificHead(nn.Module):
    """
    Task-specific output head that can be adapted for different tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        task_type: TaskType,
        num_classes: Optional[int] = None,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.task_type = task_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if task_type == TaskType.CLASSIFICATION:
            assert num_classes is not None, "num_classes required for classification"
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
        elif task_type == TaskType.REGRESSION:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )
        elif task_type == TaskType.GENERATION:
            # For generation, we'll use a transformer decoder
            self.head = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=input_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.1
                ),
                num_layers=3
            )
        elif task_type == TaskType.SEQUENCE_LABELING:
            assert num_classes is not None, "num_classes required for sequence labeling"
            self.head = nn.Linear(input_dim, num_classes)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through task-specific head."""
        if self.task_type == TaskType.GENERATION:
            # For generation tasks, additional arguments like target sequence needed
            target = kwargs.get('target')
            memory = kwargs.get('memory', x)
            return self.head(target, memory)
        else:
            return self.head(x)


class UniversalAssistant(nn.Module):
    """
    Universal AI Assistant that combines multi-modal understanding
    with few-shot learning capabilities across multiple task domains.
    """
    
    def __init__(
        self,
        backbone_config: Dict[str, Any],
        task_domains: List[str],
        meta_learning_algorithm: str = "maml",
        embed_dim: int = 512,
        **kwargs
    ):
        super().__init__()
        
        self.task_domains = task_domains
        self.embed_dim = embed_dim
        self.meta_learning_algorithm = meta_learning_algorithm
        
        # Multi-modal backbone
        # Make sure embed_dim is included in backbone_config
        if 'output_dim' not in backbone_config:
            backbone_config['output_dim'] = embed_dim
        self.backbone = MultiModalBackbone(**backbone_config)
        
        # Task-specific heads (created dynamically based on tasks)
        self.task_heads = nn.ModuleDict()
        
        # Meta-learner for few-shot adaptation
        self.meta_learner = create_meta_learner(
            algorithm=meta_learning_algorithm,
            model=self,
            **kwargs
        )
        
        # Task registry
        self.registered_tasks = {}
        
        # Current adaptation state
        self.adapted_parameters = None
        self.current_task = None
        
        logger.info(f"Initialized Universal Assistant with domains: {task_domains}")
    
    def register_task(
        self,
        task_name: str,
        task_type: TaskType,
        modality: str = "text",
        num_classes: Optional[int] = None
    ):
        """
        Register a new task with the assistant.
        
        Args:
            task_name: Unique name for the task
            task_type: Type of task (classification, regression, etc.)
            modality: Input modality (text, vision, code)
            num_classes: Number of classes for classification tasks
        """
        # Create task-specific head
        head = TaskSpecificHead(
            input_dim=self.embed_dim,
            task_type=task_type,
            num_classes=num_classes
        )
        
        self.task_heads[task_name] = head
        self.registered_tasks[task_name] = {
            'task_type': task_type,
            'modality': modality,
            'num_classes': num_classes
        }
        
        logger.info(f"Registered task: {task_name} ({task_type.value}, {modality})")
    
    def forward(
        self,
        inputs: Dict[str, Any],
        task_name: str,
        modality: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for a specific task.
        
        Args:
            inputs: Input data
            task_name: Name of the task to perform
            modality: Override modality if needed
            
        Returns:
            Task-specific outputs
        """
        if task_name not in self.registered_tasks:
            raise ValueError(f"Unknown task: {task_name}")
        
        # Determine modality
        if modality is None:
            modality = self.registered_tasks[task_name]['modality']
        
        # Encode inputs through backbone
        embeddings = self.backbone(inputs, modality=modality)
        
        # Pass through task-specific head
        task_head = self.task_heads[task_name]
        outputs = task_head(embeddings, **kwargs)
        
        return outputs
    
    def adapt(
        self,
        task_name: str,
        support_examples: List[Dict[str, Any]],
        n_shots: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Adapt the assistant to a new task using few-shot examples.
        
        Args:
            task_name: Target task name
            support_examples: List of support examples
            n_shots: Number of shots per class/example
            
        Returns:
            Adaptation results
        """
        if task_name not in self.registered_tasks:
            raise ValueError(f"Unknown task: {task_name}")
        
        # Prepare support set
        support_inputs, support_targets = self._prepare_support_set(
            support_examples, task_name, n_shots
        )
        
        # Perform meta-learning adaptation
        self.current_task = task_name
        adaptation_result = self.meta_learner.adapt(
            support_inputs, support_targets, **kwargs
        )
        
        return adaptation_result
    
    def predict(
        self,
        inputs: Dict[str, Any],
        task_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Make predictions using the (potentially adapted) model.
        
        Args:
            inputs: Input data
            task_name: Task name (uses current task if None)
            
        Returns:
            Predictions
        """
        if task_name is None:
            task_name = self.current_task
            
        if task_name is None:
            raise ValueError("No task specified and no current task set")
        
        # Use adapted parameters if available
        if self.adapted_parameters is not None:
            # Temporarily set adapted parameters
            original_params = {}
            for name, param in self.named_parameters():
                original_params[name] = param.data.clone()
                if name in self.adapted_parameters:
                    param.data = self.adapted_parameters[name]
        
        try:
            # Make prediction
            with torch.no_grad():
                predictions = self.forward(inputs, task_name, **kwargs)
        finally:
            # Restore original parameters
            if self.adapted_parameters is not None:
                for name, param in self.named_parameters():
                    if name in original_params:
                        param.data = original_params[name]
        
        return predictions
    
    def _prepare_support_set(
        self,
        support_examples: List[Dict[str, Any]],
        task_name: str,
        n_shots: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare support set for meta-learning.
        
        Args:
            support_examples: Raw support examples
            task_name: Task name
            n_shots: Number of shots
            
        Returns:
            Tuple of (support_inputs, support_targets)
        """
        task_config = self.registered_tasks[task_name]
        modality = task_config['modality']
        
        # Process examples and extract features
        support_embeddings = []
        support_targets = []
        
        for example in support_examples[:n_shots]:
            # Extract inputs and targets
            inputs = example['inputs']
            target = example['target']
            
            # Encode inputs
            with torch.no_grad():
                embedding = self.backbone(inputs, modality=modality)
                support_embeddings.append(embedding)
                support_targets.append(target)
        
        support_inputs = torch.stack(support_embeddings)
        support_targets = torch.tensor(support_targets)
        
        return support_inputs, support_targets
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'task_domains': self.task_domains,
            'embed_dim': self.embed_dim,
            'meta_learning_algorithm': self.meta_learning_algorithm,
            'registered_tasks': self.registered_tasks,
            'current_task': self.current_task
        }
        torch.save(checkpoint, path)
        logger.info(f"Universal Assistant checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.task_domains = checkpoint['task_domains']
        self.embed_dim = checkpoint['embed_dim']
        self.meta_learning_algorithm = checkpoint['meta_learning_algorithm']
        self.registered_tasks = checkpoint['registered_tasks']
        self.current_task = checkpoint.get('current_task')
        logger.info(f"Universal Assistant checkpoint loaded from {path}")


def create_universal_assistant(config: Dict[str, Any]) -> UniversalAssistant:
    """
    Factory function to create Universal Assistant from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Universal Assistant instance
    """
    assistant = UniversalAssistant(
        backbone_config=config.get('backbone', {}),
        task_domains=config.get('task_domains', ['nlp', 'vision', 'code']),
        meta_learning_algorithm=config.get('meta_algorithm', 'maml'),
        embed_dim=config.get('embed_dim', 512),
        **config.get('meta_kwargs', {})
    )
    
    # Register default tasks if specified
    default_tasks = config.get('default_tasks', {})
    for task_name, task_config in default_tasks.items():
        assistant.register_task(
            task_name=task_name,
            task_type=TaskType(task_config['type']),
            modality=task_config.get('modality', 'text'),
            num_classes=task_config.get('num_classes')
        )
    
    return assistant
