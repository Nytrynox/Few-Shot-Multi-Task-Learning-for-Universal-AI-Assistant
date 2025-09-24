"""
Training utilities and loops for meta-learning.

This module provides training loops, optimizers, and utilities
for training the Universal AI Assistant.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Callable
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path

from ..core.meta_learner import BaseMetaLearner, MultiTaskMetaLearner
from ..models.universal_assistant import UniversalAssistant
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetaTrainer:
    """
    Meta-learning trainer for the Universal AI Assistant.
    
    Handles episodic training across multiple tasks with support for
    different meta-learning algorithms.
    """
    
    def __init__(
        self,
        model: UniversalAssistant,
        meta_learner: BaseMetaLearner,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_wandb: bool = False,
        project_name: str = "universal-assistant"
    ):
        self.model = model.to(device)
        self.meta_learner = meta_learner
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.log_wandb = log_wandb
        
        # Initialize optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=0.001,
                weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Initialize wandb logging
        if self.log_wandb:
            wandb.init(project=project_name, config={
                "model": type(self.model).__name__,
                "meta_learner": type(self.meta_learner).__name__,
                "device": self.device
            })
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"MetaTrainer initialized on {device}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            leave=False
        )
        
        for batch_idx, task_batch in enumerate(progress_bar):
            # Move batch to device
            task_batch = self._move_batch_to_device(task_batch)
            
            # Perform meta-update
            meta_results = self.meta_learner.meta_update(task_batch)
            
            # Extract metrics
            meta_loss = meta_results.get('meta_loss', 0.0)
            adaptation_loss = meta_results.get('adaptation_loss', 0.0)
            accuracy = meta_results.get('accuracy', 0.0)
            
            epoch_losses.append(meta_loss)
            if accuracy > 0:
                epoch_accuracies.append(accuracy)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Meta Loss': f'{meta_loss:.4f}',
                'Adapt Loss': f'{adaptation_loss:.4f}',
                'Acc': f'{accuracy:.3f}' if accuracy > 0 else 'N/A'
            })
            
            # Log to wandb
            if self.log_wandb:
                wandb.log({
                    'train/meta_loss': meta_loss,
                    'train/adaptation_loss': adaptation_loss,
                    'train/accuracy': accuracy,
                    'train/step': self.global_step
                })
            
            self.global_step += 1
        
        # Compute epoch metrics
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies) if epoch_accuracies else 0.0
        
        # Update learning rate scheduler
        if self.scheduler:
            self.scheduler.step()
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'num_batches': len(self.train_dataloader)
        }
    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model on validation set.
        
        Args:
            dataloader: Validation dataloader (uses self.val_dataloader if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if dataloader is None:
            dataloader = self.val_dataloader
        
        if dataloader is None:
            logger.warning("No validation dataloader provided")
            return {}
        
        self.model.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for task_batch in tqdm(dataloader, desc="Validation", leave=False):
                task_batch = self._move_batch_to_device(task_batch)
                
                # Evaluate each task in the batch
                for task in task_batch:
                    support_x = task['support_x']
                    support_y = task['support_y']
                    query_x = task['query_x']
                    query_y = task['query_y']
                    task_name = task.get('task_name', 'default')
                    
                    # Adapt to support set
                    self.model.adapt(task_name, [])  # Placeholder
                    
                    # Evaluate on query set
                    predictions = self.model.predict({
                        'input_ids': query_x
                    }, task_name)
                    
                    # Compute loss and accuracy
                    if task_name in self.model.registered_tasks:
                        task_config = self.model.registered_tasks[task_name]
                        
                        if task_config['task_type'].value == 'classification':
                            loss = nn.CrossEntropyLoss()(predictions, query_y)
                            predicted_classes = torch.argmax(predictions, dim=-1)
                            accuracy = (predicted_classes == query_y).float().mean().item()
                            
                            val_losses.append(loss.item())
                            val_accuracies.append(accuracy)
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_val_accuracy = np.mean(val_accuracies) if val_accuracies else 0.0
        
        return {
            'val_loss': avg_val_loss,
            'val_accuracy': avg_val_accuracy,
            'num_episodes': len(val_losses)
        }
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            checkpoint_dir: Directory to save checkpoints
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.3f}, "
                f"Val Loss: {val_metrics.get('val_loss', 'N/A')}, "
                f"Val Acc: {val_metrics.get('val_accuracy', 'N/A')}"
            )
            
            if self.log_wandb:
                wandb.log({
                    'epoch': epoch,
                    **{f"epoch_{k}": v for k, v in train_metrics.items()},
                    **{f"epoch_{k}": v for k, v in val_metrics.items()}
                })
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(str(checkpoint_file))
            
            # Save best model
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = checkpoint_path / "best_model.pt"
                self.save_checkpoint(str(best_model_path))
                logger.info(f"New best model saved with val loss: {val_loss:.4f}")
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'registered_tasks': self.model.registered_tasks
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore registered tasks
        if 'registered_tasks' in checkpoint:
            self.model.registered_tasks = checkpoint['registered_tasks']
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def _move_batch_to_device(self, batch: List[Dict]) -> List[Dict]:
        """Move batch tensors to device."""
        device_batch = []
        
        for task in batch:
            device_task = {}
            for key, value in task.items():
                if isinstance(value, torch.Tensor):
                    device_task[key] = value.to(self.device)
                else:
                    device_task[key] = value
            device_batch.append(device_task)
        
        return device_batch


class EpisodeGenerator:
    """
    Generator for creating few-shot learning episodes from datasets.
    """
    
    def __init__(
        self,
        datasets: Dict[str, Any],
        n_way: int = 5,
        k_shot: int = 5,
        query_shots: int = 15,
        num_episodes: int = 1000
    ):
        self.datasets = datasets
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots
        self.num_episodes = num_episodes
    
    def generate_episode(self, task_name: str) -> Dict[str, torch.Tensor]:
        """
        Generate a single few-shot episode.
        
        Args:
            task_name: Name of the task to generate episode for
            
        Returns:
            Episode data containing support and query sets
        """
        if task_name not in self.datasets:
            raise ValueError(f"Dataset not found for task: {task_name}")
        
        dataset = self.datasets[task_name]
        
        # Sample classes for this episode
        available_classes = list(set([item['label'] for item in dataset]))
        if len(available_classes) < self.n_way:
            raise ValueError(f"Not enough classes in dataset {task_name}")
        
        episode_classes = np.random.choice(
            available_classes, 
            size=self.n_way, 
            replace=False
        )
        
        # Create episode data
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for class_idx, class_label in enumerate(episode_classes):
            # Get examples for this class
            class_examples = [item for item in dataset if item['label'] == class_label]
            
            if len(class_examples) < self.k_shot + self.query_shots:
                # Sample with replacement if not enough examples
                sampled_examples = np.random.choice(
                    class_examples, 
                    size=self.k_shot + self.query_shots, 
                    replace=True
                )
            else:
                sampled_examples = np.random.choice(
                    class_examples, 
                    size=self.k_shot + self.query_shots, 
                    replace=False
                )
            
            # Split into support and query
            support_examples = sampled_examples[:self.k_shot]
            query_examples = sampled_examples[self.k_shot:self.k_shot + self.query_shots]
            
            # Process examples
            for example in support_examples:
                support_x.append(self._process_example(example))
                support_y.append(class_idx)  # Use episode class index
            
            for example in query_examples:
                query_x.append(self._process_example(example))
                query_y.append(class_idx)
        
        return {
            'support_x': torch.stack(support_x) if support_x else torch.empty(0),
            'support_y': torch.tensor(support_y),
            'query_x': torch.stack(query_x) if query_x else torch.empty(0),
            'query_y': torch.tensor(query_y),
            'task_name': task_name,
            'n_way': self.n_way,
            'k_shot': self.k_shot
        }
    
    def _process_example(self, example: Dict[str, Any]) -> torch.Tensor:
        """Process a single example into tensor format."""
        # This is a simplified version - in practice, use proper preprocessing
        if 'features' in example:
            return torch.tensor(example['features'], dtype=torch.float32)
        elif 'text' in example:
            # Simple text processing
            text = example['text']
            # Convert to simple numeric representation
            return torch.tensor([hash(text) % 10000], dtype=torch.float32)
        else:
            # Default random features
            return torch.randn(128)  # 128-dimensional random features
    
    def __iter__(self):
        """Iterator interface for generating episodes."""
        for _ in range(self.num_episodes):
            # Randomly select a task
            task_name = np.random.choice(list(self.datasets.keys()))
            yield self.generate_episode(task_name)
    
    def __len__(self):
        return self.num_episodes
