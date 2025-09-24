"""
Prototypical Networks for Few-Shot Learning.

This module implements Prototypical Networks, which learn a metric space where 
classification can be performed by computing distances to prototype representations
of each class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PrototypicalNetworks:
    """
    Prototypical Networks implementation for few-shot classification.
    
    The model learns to embed support and query examples into a space where
    classification is performed by finding the nearest class prototype.
    
    Args:
        backbone: Feature extraction network
        distance_metric: Distance metric to use ('euclidean', 'cosine', 'manhattan')
        temperature: Temperature scaling for softmax
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        distance_metric: str = 'euclidean',
        temperature: float = 1.0
    ):
        self.backbone = backbone
        self.distance_metric = distance_metric
        self.temperature = temperature
        
        # Optimizer for the backbone network
        self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=0.001)
        
    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        n_classes: int
    ) -> torch.Tensor:
        """
        Compute prototype vectors for each class.
        
        Args:
            support_embeddings: Embedded support examples [n_support, embed_dim]
            support_labels: Support set labels [n_support]
            n_classes: Number of classes
            
        Returns:
            Prototype vectors [n_classes, embed_dim]
        """
        prototypes = torch.zeros(
            n_classes, 
            support_embeddings.size(-1),
            device=support_embeddings.device
        )
        
        for c in range(n_classes):
            # Find examples belonging to class c
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                # Average embeddings for this class
                prototypes[c] = support_embeddings[class_mask].mean(dim=0)
        
        return prototypes
    
    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between query embeddings and prototypes.
        
        Args:
            query_embeddings: Query embeddings [n_query, embed_dim]
            prototypes: Class prototypes [n_classes, embed_dim]
            
        Returns:
            Distance matrix [n_query, n_classes]
        """
        if self.distance_metric == 'euclidean':
            # Squared Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes, p=2) ** 2
        elif self.distance_metric == 'cosine':
            # Cosine similarity (converted to distance)
            query_norm = F.normalize(query_embeddings, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarities
        elif self.distance_metric == 'manhattan':
            # Manhattan (L1) distance
            distances = torch.cdist(query_embeddings, prototypes, p=1)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for few-shot classification.
        
        Args:
            support_x: Support set inputs [n_support, ...]
            support_y: Support set labels [n_support]
            query_x: Query set inputs [n_query, ...]
            n_classes: Number of classes in the episode
            
        Returns:
            Tuple of (logits, prototypes)
        """
        # Embed support and query examples
        support_embeddings = self.backbone(support_x)
        query_embeddings = self.backbone(query_x)
        
        # Compute class prototypes
        prototypes = self.compute_prototypes(
            support_embeddings, support_y, n_classes
        )
        
        # Compute distances and convert to logits
        distances = self.compute_distances(query_embeddings, prototypes)
        logits = -distances / self.temperature
        
        return logits, prototypes
    
    def predict(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_classes: int
    ) -> torch.Tensor:
        """
        Make predictions for query examples.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            n_classes: Number of classes
            
        Returns:
            Predicted class indices [n_query]
        """
        self.backbone.eval()
        with torch.no_grad():
            logits, _ = self.forward(support_x, support_y, query_x, n_classes)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def train_episode(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        n_classes: int
    ) -> Dict[str, float]:
        """
        Train on a single episode (task).
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels  
            query_x: Query set inputs
            query_y: Query set labels
            n_classes: Number of classes in episode
            
        Returns:
            Dictionary with loss and accuracy
        """
        self.backbone.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, prototypes = self.forward(support_x, support_y, query_x, n_classes)
        
        # Compute loss
        loss = F.cross_entropy(logits, query_y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == query_y).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'n_classes': n_classes
        }
    
    def evaluate_episode(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        n_classes: int
    ) -> Dict[str, float]:
        """
        Evaluate on a single episode without training.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            query_y: Query set labels
            n_classes: Number of classes in episode
            
        Returns:
            Dictionary with loss and accuracy
        """
        self.backbone.eval()
        with torch.no_grad():
            logits, _ = self.forward(support_x, support_y, query_x, n_classes)
            loss = F.cross_entropy(logits, query_y)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_y).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'n_classes': n_classes
        }
    
    def save_checkpoint(self, path: str):
        """Save model state."""
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'distance_metric': self.distance_metric,
            'temperature': self.temperature
        }, path)
        logger.info(f"Prototypical Networks checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.distance_metric = checkpoint['distance_metric']
        self.temperature = checkpoint['temperature']
        logger.info(f"Prototypical Networks checkpoint loaded from {path}")


class RelationNetworks:
    """
    Relation Networks for Few-Shot Learning.
    
    Instead of using fixed distance metrics, Relation Networks learn a 
    relation module that computes similarity scores between embeddings.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        relation_module: nn.Module,
        temperature: float = 1.0
    ):
        self.backbone = backbone
        self.relation_module = relation_module
        self.temperature = temperature
        
        # Optimizer for both networks
        params = list(backbone.parameters()) + list(relation_module.parameters())
        self.optimizer = torch.optim.Adam(params, lr=0.001)
    
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_classes: int
    ) -> torch.Tensor:
        """
        Forward pass using learned relation module.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            n_classes: Number of classes
            
        Returns:
            Relation scores [n_query, n_classes]
        """
        # Embed examples
        support_embeddings = self.backbone(support_x)
        query_embeddings = self.backbone(query_x)
        
        # Compute prototypes
        prototypes = torch.zeros(
            n_classes, support_embeddings.size(-1),
            device=support_embeddings.device
        )
        
        for c in range(n_classes):
            class_mask = (support_y == c)
            if class_mask.sum() > 0:
                prototypes[c] = support_embeddings[class_mask].mean(dim=0)
        
        # Compute relations between queries and prototypes
        n_query = query_embeddings.size(0)
        relations = []
        
        for i in range(n_query):
            query_embed = query_embeddings[i].unsqueeze(0)  # [1, embed_dim]
            
            # Concatenate query with each prototype
            query_proto_pairs = torch.cat([
                query_embed.repeat(n_classes, 1),  # [n_classes, embed_dim]
                prototypes  # [n_classes, embed_dim]
            ], dim=1)  # [n_classes, 2*embed_dim]
            
            # Compute relation scores
            relation_scores = self.relation_module(query_proto_pairs)
            relations.append(relation_scores.squeeze())
        
        relations = torch.stack(relations)  # [n_query, n_classes]
        return relations / self.temperature
