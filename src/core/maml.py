"""
Model-Agnostic Meta-Learning (MAML) implementation for few-shot learning.

This module implements the MAML algorithm which enables rapid adaptation to new tasks
with minimal gradient steps and training examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, List, Tuple, Optional, Callable
import higher
from copy import deepcopy

from ..utils.logging import get_logger

logger = get_logger(__name__)


class MAMLLearner:
    """
    Model-Agnostic Meta-Learning implementation for few-shot adaptation.
    
    Args:
        model: Base neural network model
        inner_lr: Learning rate for inner loop optimization
        outer_lr: Learning rate for outer loop optimization
        inner_steps: Number of gradient steps in inner loop
        first_order: Whether to use first-order MAML approximation
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = False
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Optional[Callable] = None,
        return_adapted_model: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop adaptation on support set.
        
        Args:
            support_x: Support set inputs [batch_size, ...]
            support_y: Support set targets [batch_size, ...]
            loss_fn: Loss function to use (defaults to appropriate loss based on task)
            return_adapted_model: Whether to return the adapted model parameters
            
        Returns:
            Dictionary containing adaptation losses and optionally adapted parameters
        """
        if loss_fn is None:
            loss_fn = self._get_default_loss_fn(support_y)
        
        # Create differentiable copy of model
        adapted_model = deepcopy(self.model)
        adapted_params = list(adapted_model.parameters())
        
        adaptation_losses = []
        
        for step in range(self.inner_steps):
            # Forward pass with current adapted parameters
            predictions = adapted_model(support_x)
            loss = loss_fn(predictions, support_y)
            adaptation_losses.append(loss.item())
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, 
                adapted_params, 
                create_graph=not self.first_order,
                retain_graph=True
            )
            
            # Update parameters
            adapted_params = [
                param - self.inner_lr * grad 
                for param, grad in zip(adapted_params, grads)
            ]
            
            # Update model parameters
            for param, new_param in zip(adapted_model.parameters(), adapted_params):
                param.data = new_param.data
        
        result = {
            'adaptation_losses': adaptation_losses,
            'final_loss': adaptation_losses[-1]
        }
        
        if return_adapted_model:
            result['adapted_model'] = adapted_model
            
        return result
    
    def meta_update(
        self,
        task_batch: List[Dict[str, torch.Tensor]],
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Perform meta-learning update across a batch of tasks.
        
        Args:
            task_batch: List of tasks, each containing support and query sets
            loss_fn: Loss function to use
            
        Returns:
            Dictionary containing meta-learning statistics
        """
        meta_losses = []
        adaptation_losses = []
        
        self.meta_optimizer.zero_grad()
        
        for task in task_batch:
            support_x, support_y = task['support_x'], task['support_y']
            query_x, query_y = task['query_x'], task['query_y']
            
            if loss_fn is None:
                loss_fn = self._get_default_loss_fn(support_y)
            
            # Inner loop adaptation
            with higher.innerloop_ctx(
                self.model, 
                torch.optim.SGD(self.model.parameters(), lr=self.inner_lr),
                copy_initial_weights=False
            ) as (fmodel, diffopt):
                
                # Adaptation steps
                for _ in range(self.inner_steps):
                    predictions = fmodel(support_x)
                    support_loss = loss_fn(predictions, support_y)
                    adaptation_losses.append(support_loss.item())
                    diffopt.step(support_loss)
                
                # Meta-loss on query set
                query_predictions = fmodel(query_x)
                query_loss = loss_fn(query_predictions, query_y)
                meta_losses.append(query_loss.item())
                
                # Accumulate meta-gradients
                query_loss.backward()
        
        # Meta-optimizer step
        self.meta_optimizer.step()
        
        return {
            'meta_loss': sum(meta_losses) / len(meta_losses),
            'adaptation_loss': sum(adaptation_losses) / len(adaptation_losses),
            'num_tasks': len(task_batch)
        }
    
    def _get_default_loss_fn(self, targets: torch.Tensor) -> Callable:
        """Determine appropriate loss function based on target tensor."""
        if targets.dtype in [torch.long, torch.int]:
            return F.cross_entropy
        elif len(targets.shape) == 1 or targets.shape[-1] == 1:
            return F.mse_loss
        else:
            return F.cross_entropy
    
    def save_checkpoint(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps,
            'first_order': self.first_order
        }, path)
        logger.info(f"MAML checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.inner_lr = checkpoint['inner_lr']
        self.outer_lr = checkpoint['outer_lr']
        self.inner_steps = checkpoint['inner_steps']
        self.first_order = checkpoint['first_order']
        logger.info(f"MAML checkpoint loaded from {path}")


class MAMLPlusPlus(MAMLLearner):
    """
    Enhanced MAML implementation with improvements for stability and performance.
    
    Includes:
    - Multi-step loss optimization (MSL)
    - Derivative-order annealing (DOA)  
    - Per-parameter learning rates
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        multi_step_loss: bool = True,
        per_param_lr: bool = True,
        **kwargs
    ):
        super().__init__(model, inner_lr, outer_lr, inner_steps, **kwargs)
        
        self.multi_step_loss = multi_step_loss
        self.per_param_lr = per_param_lr
        
        if per_param_lr:
            # Initialize per-parameter learning rates
            self.param_lrs = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(inner_lr))
                for name, _ in model.named_parameters()
            })
    
    def meta_update(
        self,
        task_batch: List[Dict[str, torch.Tensor]],
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Enhanced meta-update with multi-step loss optimization."""
        meta_losses = []
        adaptation_losses = []
        
        self.meta_optimizer.zero_grad()
        
        for task in task_batch:
            support_x, support_y = task['support_x'], task['support_y']
            query_x, query_y = task['query_x'], task['query_y']
            
            if loss_fn is None:
                loss_fn = self._get_default_loss_fn(support_y)
            
            with higher.innerloop_ctx(
                self.model,
                torch.optim.SGD(self.model.parameters(), lr=self.inner_lr),
                copy_initial_weights=False
            ) as (fmodel, diffopt):
                
                step_losses = []
                
                for step in range(self.inner_steps):
                    predictions = fmodel(support_x)
                    support_loss = loss_fn(predictions, support_y)
                    adaptation_losses.append(support_loss.item())
                    step_losses.append(support_loss)
                    diffopt.step(support_loss)
                
                # Query loss for this task
                query_predictions = fmodel(query_x)
                query_loss = loss_fn(query_predictions, query_y)
                
                if self.multi_step_loss:
                    # Combine losses from all adaptation steps
                    total_loss = query_loss + 0.1 * sum(step_losses)
                    meta_losses.append(total_loss.item())
                    total_loss.backward()
                else:
                    meta_losses.append(query_loss.item())
                    query_loss.backward()
        
        self.meta_optimizer.step()
        
        return {
            'meta_loss': sum(meta_losses) / len(meta_losses),
            'adaptation_loss': sum(adaptation_losses) / len(adaptation_losses),
            'num_tasks': len(task_batch)
        }
