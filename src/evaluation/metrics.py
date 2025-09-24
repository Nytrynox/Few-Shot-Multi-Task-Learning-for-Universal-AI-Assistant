"""
Evaluation metrics and utilities for the Universal AI Assistant.

This module provides comprehensive evaluation tools for assessing
few-shot learning performance across different tasks.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

from ..models.universal_assistant import UniversalAssistant
from ..tasks.base import TaskType, TaskMetrics
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FewShotEvaluator:
    """
    Evaluator for few-shot learning performance across multiple tasks.
    """
    
    def __init__(
        self,
        model: UniversalAssistant,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        
        # Evaluation results storage
        self.results = defaultdict(list)
        
    def evaluate_task(
        self,
        task_name: str,
        test_episodes: List[Dict[str, torch.Tensor]],
        n_shots_list: Optional[List[int]] = None,
        num_adaptation_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on a specific task.
        
        Args:
            task_name: Name of the task to evaluate
            test_episodes: List of test episodes
            n_shots_list: List of shot numbers to evaluate
            num_adaptation_steps: Number of adaptation steps
            
        Returns:
            Dictionary containing evaluation results
        """
        if n_shots_list is None:
            n_shots_list = [1, 5, 10]
        
        if task_name not in self.model.registered_tasks:
            raise ValueError(f"Task {task_name} not registered in model")
        
        task_config = self.model.registered_tasks[task_name]
        task_type = task_config['task_type']
        
        results = {
            'task_name': task_name,
            'task_type': task_type.value,
            'n_shots_results': {},
            'episode_results': []
        }
        
        self.model.eval()
        
        for n_shots in n_shots_list:
            shot_results = []
            
            logger.info(f"Evaluating {task_name} with {n_shots} shots")
            
            for episode_idx, episode in enumerate(test_episodes):
                episode_result = self._evaluate_episode(
                    episode, task_name, n_shots, num_adaptation_steps
                )
                shot_results.append(episode_result)
                
                if episode_idx % 50 == 0:
                    logger.info(f"Evaluated {episode_idx + 1}/{len(test_episodes)} episodes")
            
            # Aggregate results for this n_shots
            results['n_shots_results'][n_shots] = self._aggregate_episode_results(
                shot_results, task_type
            )
        
        # Store overall episode results
        results['episode_results'] = shot_results
        self.results[task_name] = results
        
        return results
    
    def _evaluate_episode(
        self,
        episode: Dict[str, torch.Tensor],
        task_name: str,
        n_shots: int,
        num_adaptation_steps: int
    ) -> Dict[str, Any]:
        """
        Evaluate a single episode.
        
        Args:
            episode: Episode data
            task_name: Task name
            n_shots: Number of shots for support set
            num_adaptation_steps: Number of adaptation steps
            
        Returns:
            Episode evaluation results
        """
        # Move episode to device
        episode = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in episode.items()}
        
        support_x = episode['support_x'][:n_shots]
        support_y = episode['support_y'][:n_shots]
        query_x = episode['query_x']
        query_y = episode['query_y']
        
        # Prepare support examples for adaptation
        support_examples = []
        for i in range(len(support_x)):
            support_examples.append({
                'inputs': {'input_ids': support_x[i]},
                'target': support_y[i].item()
            })
        
        # Adapt model to support set
        with torch.no_grad():
            adaptation_result = self.model.adapt(
                task_name=task_name,
                support_examples=support_examples,
                n_shots=n_shots
            )
        
        # Make predictions on query set
        query_predictions = []
        query_targets = query_y.cpu().numpy()
        
        for i in range(len(query_x)):
            query_input = {'input_ids': query_x[i].unsqueeze(0)}
            
            with torch.no_grad():
                prediction = self.model.predict(query_input, task_name)
                
                # Convert prediction to appropriate format
                if len(prediction.shape) > 1 and prediction.shape[-1] > 1:
                    # Classification: take argmax
                    pred_class = torch.argmax(prediction, dim=-1).cpu().numpy()[0]
                    query_predictions.append(pred_class)
                else:
                    # Regression: take raw value
                    pred_value = prediction.cpu().numpy()[0]
                    query_predictions.append(pred_value)
        
        # Compute metrics
        task_config = self.model.registered_tasks[task_name]
        task_type = task_config['task_type']
        
        if task_type == TaskType.CLASSIFICATION:
            accuracy = TaskMetrics.accuracy(query_predictions, query_targets.tolist())
            f1 = TaskMetrics.f1_score(query_predictions, query_targets.tolist())
            
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'num_correct': sum(1 for p, t in zip(query_predictions, query_targets) if p == t),
                'num_total': len(query_predictions)
            }
        elif task_type == TaskType.REGRESSION:
            mse = TaskMetrics.mse(query_predictions, query_targets.tolist())
            mae = TaskMetrics.mae(query_predictions, query_targets.tolist())
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'num_samples': len(query_predictions)
            }
        else:
            # Default metrics
            metrics = {
                'num_samples': len(query_predictions)
            }
        
        return {
            'n_shots': n_shots,
            'adaptation_loss': adaptation_result.get('final_loss', 0.0),
            'metrics': metrics,
            'predictions': query_predictions,
            'targets': query_targets.tolist()
        }
    
    def _aggregate_episode_results(
        self,
        episode_results: List[Dict[str, Any]],
        task_type: TaskType
    ) -> Dict[str, float]:
        """
        Aggregate results across episodes.
        
        Args:
            episode_results: List of episode results
            task_type: Type of task
            
        Returns:
            Aggregated metrics
        """
        if not episode_results:
            return {}
        
        aggregated = {}
        
        if task_type == TaskType.CLASSIFICATION:
            accuracies = [r['metrics']['accuracy'] for r in episode_results]
            f1_scores = [r['metrics']['f1_score'] for r in episode_results]
            
            aggregated['mean_accuracy'] = np.mean(accuracies)
            aggregated['std_accuracy'] = np.std(accuracies)
            aggregated['mean_f1_score'] = np.mean(f1_scores)
            aggregated['std_f1_score'] = np.std(f1_scores)
            
            # Confidence intervals (95%)
            aggregated['accuracy_ci'] = 1.96 * aggregated['std_accuracy'] / np.sqrt(len(accuracies))
            aggregated['f1_ci'] = 1.96 * aggregated['std_f1_score'] / np.sqrt(len(f1_scores))
            
        elif task_type == TaskType.REGRESSION:
            mse_values = [r['metrics']['mse'] for r in episode_results]
            mae_values = [r['metrics']['mae'] for r in episode_results]
            
            aggregated['mean_mse'] = np.mean(mse_values)
            aggregated['std_mse'] = np.std(mse_values)
            aggregated['mean_mae'] = np.mean(mae_values)
            aggregated['std_mae'] = np.std(mae_values)
        
        # Common metrics
        adaptation_losses = [r['adaptation_loss'] for r in episode_results]
        aggregated['mean_adaptation_loss'] = np.mean(adaptation_losses)
        aggregated['std_adaptation_loss'] = np.std(adaptation_losses)
        aggregated['num_episodes'] = len(episode_results)
        
        return aggregated
    
    def evaluate_multiple_tasks(
        self,
        task_episodes: Dict[str, List[Dict[str, torch.Tensor]]],
        n_shots_list: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on multiple tasks.
        
        Args:
            task_episodes: Dictionary mapping task names to episode lists
            n_shots_list: List of shot numbers to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        if n_shots_list is None:
            n_shots_list = [1, 5, 10]
        
        all_results = {}
        
        for task_name, episodes in task_episodes.items():
            logger.info(f"Evaluating task: {task_name}")
            task_results = self.evaluate_task(task_name, episodes, n_shots_list)
            all_results[task_name] = task_results
        
        # Compute cross-task statistics
        cross_task_results = self._compute_cross_task_metrics(all_results, n_shots_list)
        
        return {
            'task_results': all_results,
            'cross_task_results': cross_task_results,
            'evaluation_config': {
                'n_shots_list': n_shots_list,
                'num_tasks': len(task_episodes),
                'device': self.device
            }
        }
    
    def _compute_cross_task_metrics(
        self,
        all_results: Dict[str, Any],
        n_shots_list: List[int]
    ) -> Dict[str, Any]:
        """Compute metrics aggregated across all tasks."""
        cross_task = {
            'n_shots_results': {},
            'task_summary': {}
        }
        
        for n_shots in n_shots_list:
            task_accuracies = []
            task_f1_scores = []
            
            for task_name, task_results in all_results.items():
                shot_results = task_results['n_shots_results'].get(n_shots, {})
                
                if 'mean_accuracy' in shot_results:
                    task_accuracies.append(shot_results['mean_accuracy'])
                if 'mean_f1_score' in shot_results:
                    task_f1_scores.append(shot_results['mean_f1_score'])
            
            cross_task['n_shots_results'][n_shots] = {
                'mean_accuracy_across_tasks': np.mean(task_accuracies) if task_accuracies else None,
                'std_accuracy_across_tasks': np.std(task_accuracies) if task_accuracies else None,
                'mean_f1_across_tasks': np.mean(task_f1_scores) if task_f1_scores else None,
                'std_f1_across_tasks': np.std(task_f1_scores) if task_f1_scores else None,
                'num_tasks': len(task_accuracies)
            }
        
        # Task-wise summary
        for task_name, task_results in all_results.items():
            task_type = task_results['task_type']
            best_accuracy = 0
            
            for n_shots, shot_results in task_results['n_shots_results'].items():
                accuracy = shot_results.get('mean_accuracy', 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
            
            cross_task['task_summary'][task_name] = {
                'task_type': task_type,
                'best_accuracy': best_accuracy
            }
        
        return cross_task
    
    def save_results(self, output_path: str):
        """Save evaluation results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(dict(self.results))
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def load_results(self, input_path: str):
        """Load evaluation results from file."""
        with open(input_path, 'r') as f:
            loaded_results = json.load(f)
        
        self.results = defaultdict(list, loaded_results)
        logger.info(f"Evaluation results loaded from {input_path}")
    
    def print_summary(self):
        """Print a summary of evaluation results."""
        if not self.results:
            logger.info("No evaluation results to display")
            return
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        for task_name, task_results in self.results.items():
            print(f"\nTask: {task_name} ({task_results['task_type']})")
            print("-" * 50)
            
            for n_shots, shot_results in task_results['n_shots_results'].items():
                print(f"\n{n_shots}-shot performance:")
                
                if 'mean_accuracy' in shot_results:
                    accuracy = shot_results['mean_accuracy']
                    accuracy_ci = shot_results.get('accuracy_ci', 0)
                    print(f"  Accuracy: {accuracy:.3f} ± {accuracy_ci:.3f}")
                
                if 'mean_f1_score' in shot_results:
                    f1 = shot_results['mean_f1_score']
                    f1_ci = shot_results.get('f1_ci', 0)
                    print(f"  F1 Score: {f1:.3f} ± {f1_ci:.3f}")
                
                if 'mean_mse' in shot_results:
                    mse = shot_results['mean_mse']
                    print(f"  MSE: {mse:.4f}")
                
                if 'mean_mae' in shot_results:
                    mae = shot_results['mean_mae']
                    print(f"  MAE: {mae:.4f}")
                
                num_episodes = shot_results.get('num_episodes', 0)
                print(f"  Episodes: {num_episodes}")


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for evaluating Universal AI Assistants.
    """
    
    def __init__(
        self,
        model: UniversalAssistant,
        benchmark_config: Dict[str, Any]
    ):
        self.model = model
        self.config = benchmark_config
        self.evaluator = FewShotEvaluator(model)
        
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the full benchmark suite.
        
        Returns:
            Comprehensive benchmark results
        """
        logger.info("Starting benchmark evaluation")
        
        # Load benchmark datasets
        benchmark_datasets = self._load_benchmark_datasets()
        
        # Generate test episodes
        test_episodes = self._generate_test_episodes(benchmark_datasets)
        
        # Run evaluation
        results = self.evaluator.evaluate_multiple_tasks(
            test_episodes,
            n_shots_list=self.config.get('n_shots_list', [1, 5, 10])
        )
        
        # Add benchmark metadata
        results['benchmark_info'] = {
            'version': self.config.get('version', '1.0'),
            'model_name': type(self.model).__name__,
            'tasks_evaluated': list(benchmark_datasets.keys()),
            'total_episodes': sum(len(episodes) for episodes in test_episodes.values())
        }
        
        logger.info("Benchmark evaluation completed")
        return results
    
    def _load_benchmark_datasets(self) -> Dict[str, Any]:
        """Load benchmark datasets."""
        # Placeholder implementation
        # In practice, load standard benchmark datasets like Mini-ImageNet, Omniglot, etc.
        return {}
    
    def _generate_test_episodes(self, datasets: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Generate test episodes from benchmark datasets."""
        # Placeholder implementation
        return {}
