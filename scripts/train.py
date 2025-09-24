"""
Main training script for the Universal AI Assistant.

This script handles the complete training pipeline including data loading,
model initialization, meta-learning training, and evaluation.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.universal_assistant import create_universal_assistant
from src.core.meta_learner import create_meta_learner
from src.training.meta_trainer import MetaTrainer, EpisodeGenerator  
from src.evaluation.metrics import FewShotEvaluator
from src.tasks.base import TaskType
from src.utils.logging import configure_logging, get_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: dict) -> str:
    """Setup and return the appropriate device."""
    device_config = config.get('hardware', {}).get('device', 'auto')
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = device_config
    
    print(f"Using device: {device}")
    return device


def create_dummy_datasets(config: dict) -> dict:
    """
    Create dummy datasets for demonstration.
    In production, replace with actual dataset loading.
    """
    datasets = {}
    
    # Create dummy NLP dataset
    nlp_data = []
    for i in range(1000):
        nlp_data.append({
            'text': f'This is sample text {i} for sentiment analysis.',
            'label': i % 2,  # Binary classification
            'features': torch.randn(128).tolist()
        })
    datasets['sentiment_analysis'] = nlp_data
    
    # Create dummy topic classification dataset
    topic_data = []
    for i in range(1000):
        topic_data.append({
            'text': f'Topic {i % 10} related content goes here.',
            'label': i % 10,  # 10-class classification
            'features': torch.randn(128).tolist()
        })
    datasets['topic_classification'] = topic_data
    
    # Create dummy vision dataset
    vision_data = []
    for i in range(1000):
        vision_data.append({
            'image': torch.randn(3, 224, 224),
            'label': i % 5,  # 5-class classification
            'features': torch.randn(128).tolist()
        })
    datasets['image_classification'] = vision_data
    
    return datasets


def create_data_loaders(config: dict, datasets: dict) -> tuple:
    """Create training and validation data loaders."""
    training_config = config.get('training', {})
    episode_config = training_config.get('episodes', {})
    
    # Create episode generators
    train_generator = EpisodeGenerator(
        datasets=datasets,
        n_way=episode_config.get('n_way', 5),
        k_shot=episode_config.get('k_shot', 5),
        query_shots=episode_config.get('query_shots', 15),
        num_episodes=episode_config.get('num_train_episodes', 1000)
    )
    
    val_generator = EpisodeGenerator(
        datasets=datasets,
        n_way=episode_config.get('n_way', 5),
        k_shot=episode_config.get('k_shot', 5),
        query_shots=episode_config.get('query_shots', 15),
        num_episodes=episode_config.get('num_val_episodes', 200)
    )
    
    # Convert to data loaders
    from torch.utils.data import DataLoader
    
    # Create simple dataset wrapper
    class EpisodeDataset:
        def __init__(self, generator):
            self.generator = generator
            self.episodes = list(generator)
        
        def __len__(self):
            return len(self.episodes)
        
        def __getitem__(self, idx):
            return self.episodes[idx]
    
    train_dataset = EpisodeDataset(train_generator)
    val_dataset = EpisodeDataset(val_generator)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get('batch_size', 4),
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('batch_size', 4),
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train Universal AI Assistant")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/universal_assistant.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--eval_only', 
        action='store_true',
        help='Only run evaluation, do not train'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logging_config = config.get('logging', {})
    configure_logging(
        console_level=logging_config.get('level', 'INFO'),
        file_level='DEBUG',
        log_file=logging_config.get('log_file')
    )
    
    logger = get_logger(__name__)
    logger.info("Starting Universal AI Assistant training")
    
    # Setup device
    device = setup_device(config)
    
    # Create model
    logger.info("Creating Universal AI Assistant model")
    assistant = create_universal_assistant(config)
    
    # Load datasets
    logger.info("Loading datasets")
    datasets = create_dummy_datasets(config)
    
    # Create data loaders
    logger.info("Creating data loaders")
    train_loader, val_loader = create_data_loaders(config, datasets)
    
    if not args.eval_only:
        # Create trainer
        logger.info("Setting up trainer")
        
        # Create optimizer
        training_config = config.get('training', {})
        optimizer = torch.optim.Adam(
            assistant.parameters(),
            lr=training_config.get('learning_rate', 0.001),
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
        
        # Create meta-learner
        meta_learner = create_meta_learner(
            algorithm=config['model']['meta_algorithm'],
            model=assistant,
            **config['model'].get('meta_kwargs', {})
        )
        
        # Create trainer
        trainer = MetaTrainer(
            model=assistant,
            meta_learner=meta_learner,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_wandb=config.get('logging', {}).get('wandb', {}).get('enabled', False),
            project_name=config.get('logging', {}).get('wandb', {}).get('project', 'universal-assistant')
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Train the model
        logger.info("Starting training")
        trainer.train(
            num_epochs=training_config.get('num_epochs', 100),
            save_every=training_config.get('save_every', 10),
            checkpoint_dir=training_config.get('checkpoint_dir', './checkpoints')
        )
    
    # Evaluation
    logger.info("Starting evaluation")
    
    # Create evaluator
    evaluator = FewShotEvaluator(assistant, device=device)
    
    # Generate test episodes
    eval_config = config.get('evaluation', {})
    test_generator = EpisodeGenerator(
        datasets=datasets,
        n_way=5,
        k_shot=5,
        query_shots=15,
        num_episodes=eval_config.get('num_test_episodes', 200)
    )
    
    # Create test episodes for each task
    test_episodes = {}
    for task_name in datasets.keys():
        if task_name in assistant.registered_tasks:
            task_episodes = []
            for _ in range(eval_config.get('num_test_episodes', 200) // len(datasets)):
                episode = test_generator.generate_episode(task_name)
                task_episodes.append(episode)
            test_episodes[task_name] = task_episodes
    
    # Run evaluation
    results = evaluator.evaluate_multiple_tasks(
        test_episodes,
        n_shots_list=eval_config.get('n_shots_list', [1, 5, 10])
    )
    
    # Print and save results
    evaluator.print_summary()
    
    # Save results
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    evaluator.save_results('./results/evaluation_results.json')
    
    logger.info("Training and evaluation completed!")


if __name__ == '__main__':
    main()
