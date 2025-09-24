"""
Natural Language Processing task implementations.

This module contains concrete implementations of various NLP tasks
for the Universal AI Assistant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import Counter

from .base import BaseTask, TaskType, TaskDomain, Modality, TaskFactory, TaskMetrics


@TaskFactory.register_task
class TextClassification(BaseTask):
    """
    Text classification task implementation.
    
    Supports sentiment analysis, topic classification, spam detection, etc.
    """
    
    def __init__(
        self,
        name: str = "text_classification",
        num_classes: int = 2,
        max_length: int = 512,
        tokenizer_name: str = "bert-base-uncased"
    ):
        super().__init__(
            name=name,
            task_type=TaskType.CLASSIFICATION,
            domain=TaskDomain.NLP,
            modality=Modality.TEXT,
            description="Text classification task for sentiment analysis, topic classification, etc."
        )
        
        self.num_classes = num_classes
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        
        # Simple word-based tokenizer (replace with proper tokenizer in production)
        self.vocab = {}
        self.vocab_size = 0
    
    def preprocess_input(self, raw_input: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text input for the model.
        
        Args:
            raw_input: Raw text string
            
        Returns:
            Dictionary containing tokenized and encoded text
        """
        # Simple tokenization (replace with proper tokenizer)
        text = raw_input.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = text.split()
        
        # Convert to IDs (simplified - use proper tokenizer in practice)
        token_ids = []
        for token in tokens[:self.max_length]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            token_ids.append(self.vocab[token])
        
        # Pad to max_length
        while len(token_ids) < self.max_length:
            token_ids.append(0)  # Padding token
        
        return {
            'input_ids': torch.tensor(token_ids[:self.max_length], dtype=torch.long),
            'attention_mask': torch.tensor([1] * min(len(tokens), self.max_length) + 
                                         [0] * max(0, self.max_length - len(tokens)), dtype=torch.long),
            'text': raw_input
        }
    
    def postprocess_output(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """
        Convert model logits to classification results.
        
        Args:
            model_output: Model logits [batch_size, num_classes]
            
        Returns:
            Classification results with probabilities and predicted class
        """
        probabilities = F.softmax(model_output, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        
        return {
            'predicted_class': predicted_class.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'confidence': torch.max(probabilities, dim=-1)[0].cpu().numpy()
        }
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss for classification."""
        return F.cross_entropy(predictions, targets)
    
    def evaluate(
        self,
        predictions: List[int],
        targets: List[int]
    ) -> Dict[str, float]:
        """Evaluate classification performance."""
        return {
            'accuracy': TaskMetrics.accuracy(predictions, targets),
            'f1_score': TaskMetrics.f1_score(predictions, targets)
        }


@TaskFactory.register_task
class TextGeneration(BaseTask):
    """
    Text generation task implementation.
    
    Supports story generation, dialogue generation, etc.
    """
    
    def __init__(
        self,
        name: str = "text_generation",
        max_length: int = 512,
        vocab_size: int = 50000
    ):
        super().__init__(
            name=name,
            task_type=TaskType.GENERATION,
            domain=TaskDomain.NLP,
            modality=Modality.TEXT,
            description="Text generation task for story generation, dialogue, etc."
        )
        
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.vocab = {}
    
    def preprocess_input(self, raw_input: str) -> Dict[str, torch.Tensor]:
        """Preprocess text input for generation."""
        # Simple tokenization
        text = raw_input.lower()
        tokens = text.split()
        
        token_ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            token_ids.append(self.vocab[token])
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'text': raw_input
        }
    
    def postprocess_output(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to generated text."""
        # Apply softmax and sample
        probabilities = F.softmax(model_output, dim=-1)
        generated_ids = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), 1)
        
        # Convert back to text (simplified)
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        generated_tokens = [reverse_vocab.get(id.item(), '<unk>') for id in generated_ids]
        generated_text = ' '.join(generated_tokens)
        
        return {
            'generated_text': generated_text,
            'generated_ids': generated_ids.cpu().numpy()
        }
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss for generation."""
        # Shift targets for next token prediction
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-1
        )
    
    def evaluate(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Evaluate generation quality."""
        return {
            'bleu_score': TaskMetrics.bleu_score(predictions, targets)
        }


@TaskFactory.register_task
class QuestionAnswering(BaseTask):
    """
    Question answering task implementation.
    
    Supports extractive and abstractive QA.
    """
    
    def __init__(
        self,
        name: str = "question_answering",
        max_context_length: int = 512,
        max_question_length: int = 128,
        answer_type: str = "extractive"  # or "generative"
    ):
        super().__init__(
            name=name,
            task_type=TaskType.QUESTION_ANSWERING,
            domain=TaskDomain.NLP,
            modality=Modality.TEXT,
            description="Question answering task for reading comprehension"
        )
        
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
        self.answer_type = answer_type
        self.vocab = {}
    
    def preprocess_input(self, raw_input: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Preprocess QA input containing question and context.
        
        Args:
            raw_input: Dictionary with 'question' and 'context' keys
            
        Returns:
            Processed input tensors
        """
        question = raw_input['question'].lower()
        context = raw_input['context'].lower()
        
        # Simple tokenization
        question_tokens = question.split()[:self.max_question_length]
        context_tokens = context.split()[:self.max_context_length]
        
        # Combine question and context
        combined_tokens = question_tokens + ['[SEP]'] + context_tokens
        
        # Convert to IDs
        token_ids = []
        for token in combined_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            token_ids.append(self.vocab[token])
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'question': raw_input['question'],
            'context': raw_input['context'],
            'question_length': len(question_tokens),
            'context_start': len(question_tokens) + 1  # After [SEP]
        }
    
    def postprocess_output(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to answer."""
        if self.answer_type == "extractive":
            # For extractive QA, model outputs start and end positions
            start_logits, end_logits = model_output.chunk(2, dim=-1)
            start_pos = torch.argmax(start_logits, dim=-1)
            end_pos = torch.argmax(end_logits, dim=-1)
            
            return {
                'answer_start': start_pos.cpu().numpy(),
                'answer_end': end_pos.cpu().numpy(),
                'start_logits': start_logits.cpu().numpy(),
                'end_logits': end_logits.cpu().numpy()
            }
        else:
            # For generative QA, treat as text generation
            probabilities = F.softmax(model_output, dim=-1)
            generated_ids = torch.argmax(probabilities, dim=-1)
            
            return {
                'generated_answer_ids': generated_ids.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy()
            }
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for QA task."""
        if self.answer_type == "extractive":
            # Targets should contain start and end positions
            start_logits, end_logits = predictions.chunk(2, dim=-1)
            start_targets, end_targets = targets.chunk(2, dim=-1)
            
            start_loss = F.cross_entropy(start_logits.squeeze(-1), start_targets.squeeze(-1))
            end_loss = F.cross_entropy(end_logits.squeeze(-1), end_targets.squeeze(-1))
            
            return (start_loss + end_loss) / 2
        else:
            return F.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1))
    
    def evaluate(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Evaluate QA performance."""
        # Compute exact match and F1 score
        exact_matches = 0
        f1_scores = []
        
        for pred, target in zip(predictions, targets):
            # Exact match
            if pred.strip().lower() == target.strip().lower():
                exact_matches += 1
            
            # Token-level F1
            pred_tokens = set(pred.lower().split())
            target_tokens = set(target.lower().split())
            
            if len(pred_tokens) == 0 and len(target_tokens) == 0:
                f1_scores.append(1.0)
            elif len(pred_tokens) == 0 or len(target_tokens) == 0:
                f1_scores.append(0.0)
            else:
                common_tokens = pred_tokens & target_tokens
                precision = len(common_tokens) / len(pred_tokens)
                recall = len(common_tokens) / len(target_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
        
        return {
            'exact_match': exact_matches / len(predictions) if predictions else 0,
            'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0
        }


@TaskFactory.register_task  
class TextSummarization(BaseTask):
    """
    Text summarization task implementation.
    """
    
    def __init__(
        self,
        name: str = "text_summarization",
        max_input_length: int = 1024,
        max_summary_length: int = 256
    ):
        super().__init__(
            name=name,
            task_type=TaskType.SUMMARIZATION,
            domain=TaskDomain.NLP,
            modality=Modality.TEXT,
            description="Text summarization task for document summarization"
        )
        
        self.max_input_length = max_input_length
        self.max_summary_length = max_summary_length
        self.vocab = {}
    
    def preprocess_input(self, raw_input: str) -> Dict[str, torch.Tensor]:
        """Preprocess document for summarization."""
        text = raw_input.lower()
        tokens = text.split()[:self.max_input_length]
        
        token_ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            token_ids.append(self.vocab[token])
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'text': raw_input
        }
    
    def postprocess_output(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to summary."""
        probabilities = F.softmax(model_output, dim=-1)
        summary_ids = torch.argmax(probabilities, dim=-1)
        
        # Convert back to text
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        summary_tokens = [reverse_vocab.get(id.item(), '<unk>') for id in summary_ids]
        summary_text = ' '.join(summary_tokens)
        
        return {
            'summary': summary_text,
            'summary_ids': summary_ids.cpu().numpy()
        }
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for summarization."""
        return F.cross_entropy(
            predictions.view(-1, predictions.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
    
    def evaluate(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Evaluate summarization quality."""
        return {
            'bleu_score': TaskMetrics.bleu_score(predictions, targets)
        }
