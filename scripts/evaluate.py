"""
Evaluation script for the Universal AI Assistant.

This script runs comprehensive evaluation of the Universal AI Assistant
across multiple tasks and few-shot scenarios.
"""

import argparse
import yaml
import torch
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.universal_assistant import create_universal_assistant
from src.evaluation.metrics import FewShotEvaluator, BenchmarkSuite
from src.training.meta_trainer import EpisodeGenerator
from src.utils.logging import configure_logging, get_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_test_datasets(config: dict) -> dict:
    """Create test datasets for evaluation."""
    datasets = {}
    
    # NLP datasets
    datasets['sentiment_analysis'] = []
    for i in range(500):
        datasets['sentiment_analysis'].append({
            'text': f'Sample sentiment text {i}',
            'label': i % 2,
            'features': torch.randn(128).tolist()
        })
    
    datasets['topic_classification'] = []
    for i in range(500):
        datasets['topic_classification'].append({
            'text': f'Sample topic text {i}',
            'label': i % 5,
            'features': torch.randn(128).tolist()
        })
    
    # Vision dataset
    datasets['image_classification'] = []
    for i in range(500):
        datasets['image_classification'].append({
            'image': torch.randn(3, 224, 224),
            'label': i % 10,
            'features': torch.randn(128).tolist()
        })
    
    return datasets


def generate_test_episodes(datasets: dict, config: dict) -> dict:
    """Generate test episodes for evaluation."""
    eval_config = config.get('evaluation', {})
    
    test_episodes = {}
    
    for task_name in datasets.keys():
        episode_generator = EpisodeGenerator(
            datasets={task_name: datasets[task_name]},
            n_way=5,
            k_shot=5,
            query_shots=15,
            num_episodes=eval_config.get('num_test_episodes', 100)
        )
        
        episodes = []
        for _ in range(eval_config.get('num_test_episodes', 100)):
            episode = episode_generator.generate_episode(task_name)
            episodes.append(episode)
        
        test_episodes[task_name] = episodes
    
    return test_episodes


def main():
    parser = argparse.ArgumentParser(description="Evaluate Universal AI Assistant")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/universal_assistant.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--n_shots',
        type=int,
        nargs='+',
        default=[1, 5, 10],
        help='List of shot numbers to evaluate'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    configure_logging(console_level='INFO')
    logger = get_logger(__name__)
    
    logger.info("Starting Universal AI Assistant evaluation")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating Universal AI Assistant model")
    assistant = create_universal_assistant(config)
    assistant = assistant.to(device)
    
    # Load trained model if provided
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        try:
            assistant.load_checkpoint(args.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            logger.info("Proceeding with randomly initialized model")
    
    # Create test datasets
    logger.info("Creating test datasets")
    test_datasets = create_test_datasets(config)
    
    # Generate test episodes
    logger.info("Generating test episodes")
    test_episodes = generate_test_episodes(test_datasets, config)
    
    logger.info(f"Generated episodes for tasks: {list(test_episodes.keys())}")
    for task_name, episodes in test_episodes.items():
        logger.info(f"  {task_name}: {len(episodes)} episodes")
    
    # Create evaluator
    logger.info("Creating evaluator")
    evaluator = FewShotEvaluator(assistant, device=device)
    
    # Run evaluation
    logger.info("Starting evaluation")
    results = evaluator.evaluate_multiple_tasks(
        test_episodes,
        n_shots_list=args.n_shots
    )
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / 'evaluation_results.json'
    evaluator.save_results(str(results_file))
    
    # Save summary statistics
    summary_file = output_dir / 'evaluation_summary.json'
    summary_stats = {
        'config': config,
        'evaluation_args': vars(args),
        'cross_task_results': results['cross_task_results'],
        'device': device,
        'model_path': args.model_path
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("Evaluation completed!")
    
    # Print key metrics
    cross_task = results.get('cross_task_results', {})
    if cross_task:
        logger.info("\n" + "="*50)
        logger.info("KEY EVALUATION METRICS")
        logger.info("="*50)
        
        for n_shots in args.n_shots:
            shot_results = cross_task.get('n_shots_results', {}).get(n_shots, {})
            if shot_results:
                mean_acc = shot_results.get('mean_accuracy_across_tasks')
                if mean_acc is not None:
                    logger.info(f"{n_shots}-shot average accuracy: {mean_acc:.3f}")
    
    return results


if __name__ == '__main__':
    main()
