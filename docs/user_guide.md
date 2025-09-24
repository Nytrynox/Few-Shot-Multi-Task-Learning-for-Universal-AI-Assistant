# Universal AI Assistant: Complete User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Command Line Interface](#command-line-interface)
5. [Graphical User Interface](#graphical-user-interface)
6. [Working with Tasks](#working-with-tasks)
7. [Few-Shot Learning Examples](#few-shot-learning-examples)
8. [Model Training and Fine-tuning](#model-training-and-fine-tuning)
9. [Performance Analysis](#performance-analysis)
10. [Advanced Usage](#advanced-usage)
11. [Troubleshooting](#troubleshooting)
12. [API Reference](#api-reference)

## Introduction

The Universal AI Assistant is a powerful few-shot multi-task learning framework that enables rapid adaptation to new tasks with minimal examples. Built on meta-learning principles, it can perform across diverse domains including natural language processing, computer vision, code generation, and mathematical reasoning.

Key capabilities include:
- Adapting to new tasks with just a few examples (few-shot learning)
- Transferring knowledge across multiple task domains
- Learning to learn through meta-learning algorithms
- Processing multiple modalities (text, images, code)

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Pip package manager

### Step-by-Step Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vishnu
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Verify installation:
   ```bash
   python -c "from src.models.universal_assistant import UniversalAssistant; print('Installation successful!')"
   ```

### GPU Support (Recommended)

For optimal performance, we recommend using a CUDA-enabled GPU:

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Getting Started

### Quick Demo

To quickly see the assistant in action:

```bash
# Run the GUI for an interactive experience
chmod +x run_gui.sh
./run_gui.sh
```

Or try the CLI:

```bash
python src/cli.py --interactive
```

## Command Line Interface

The CLI provides a text-based interface for interacting with the assistant.

### Launch the CLI

```bash
python src/cli.py --interactive
```

### Available Commands

- `list` - Show all registered tasks
- `register <name> <type> <modality> [classes]` - Register a new task
- `adapt <task> <text>` - Adapt the model to a task with an example
- `predict <task> <text>` - Make a prediction using the adapted model
- `help` - Show the help menu
- `quit` - Exit the CLI

### Examples

```
> list
Registered Tasks:
----------------------------------------
• sentiment_analysis
  Type: classification
  Modality: text
  Classes: 2

> adapt sentiment_analysis "I loved this movie! It was fantastic." positive
Adapting to task 'sentiment_analysis' with example...
Adaptation complete!

> predict sentiment_analysis "The movie was terrible and boring"
Prediction: negative (confidence: 0.92)
```

## Graphical User Interface

The GUI provides an intuitive, visual interface for interacting with the assistant.

### Launch the GUI

```bash
chmod +x run_gui.sh
./run_gui.sh
```

The web interface will automatically open in your default browser at http://localhost:8501.

### GUI Components

#### 1. Home
- Overview of the assistant
- System status and loaded models
- Quick links to common tasks

#### 2. Task Demonstration
- Interactive task testing
- Few-shot example addition
- Real-time prediction visualization

#### 3. Task Explorer
- Browse all registered tasks
- Create new tasks
- Edit task parameters
- Delete tasks

#### 4. Model Analysis
- Performance visualizations
- Adaptation metrics
- Embedding space projections
- Confidence analysis

#### 5. Settings
- Configure model parameters
- Adjust UI preferences
- Manage system resources

### Using the GUI for Few-Shot Learning

1. Go to the **Task Demonstration** tab
2. Select a task type (e.g., "sentiment_analysis")
3. Add few-shot examples by providing input text and expected output
4. Test the adapted model with new inputs
5. Visualize learning progress in real-time

## Working with Tasks

Tasks are the fundamental units of work for the assistant. Each task has:
- A name
- A task type (classification, generation, etc.)
- A modality (text, vision, code)
- Optional parameters (like number of classes for classification)

### Task Types

1. **Classification Tasks**
   - Sentiment analysis
   - Image classification
   - Intent recognition

2. **Generation Tasks**
   - Text completion
   - Code generation
   - Summarization

3. **Regression Tasks**
   - Score prediction
   - Value estimation

### Registering a New Task

#### Via CLI:
```
register topic_classification classification text 5
```

#### Via GUI:
1. Go to "Task Explorer"
2. Click "New Task"
3. Fill in the task details
4. Click "Create"

### Task Adaptation Process

1. Select a task
2. Provide examples (input + expected output)
3. The meta-learner adapts to the task
4. The adapted model can now make predictions

## Few-Shot Learning Examples

### Text Classification Example

```python
# 1. Register task
assistant.register_task("news_classification", "classification", "text", classes=4)

# 2. Provide few-shot examples
examples = [
    ("Scientists discover new species in Amazon rainforest", "Science"),
    ("Stock market reaches all-time high", "Business"),
    ("New movie breaks box office records", "Entertainment")
]

# 3. Adapt the model
for text, label in examples:
    assistant.adapt("news_classification", text, label)

# 4. Make predictions
result = assistant.predict("news_classification", "New vaccine shows promising results")
# Output: "Science" (with confidence score)
```

### Image Classification Example

Through the GUI:
1. Select "image_classification" task
2. Upload 2-3 example images per class
3. Test with new images

## Model Training and Fine-tuning

For advanced users who want to train or fine-tune the models:

### Meta-Training

```bash
python scripts/meta_train.py --config configs/meta_training.yaml
```

Key configuration parameters:
- `meta_lr`: Meta-learning rate
- `inner_lr`: Inner adaptation learning rate
- `n_shots`: Number of examples per task
- `n_ways`: Number of classes per task
- `tasks`: Task domains to include

### Evaluation

```bash
python scripts/evaluate.py --model_path checkpoints/best_model.pt --test_tasks data/test_tasks.json
```

## Performance Analysis

The GUI provides comprehensive visualizations for analyzing model performance:

### Available Visualizations

1. **Learning Curves**
   - Shows how performance improves with more examples

2. **Embedding Space Projections**
   - Visualizes how the model represents different tasks

3. **Confidence Analysis**
   - Assesses model certainty across different inputs

4. **Cross-Task Transfer**
   - Measures how learning one task impacts performance on others

5. **Adaptation Speed**
   - Shows how quickly the model adapts to new tasks

## Advanced Usage

### Using the Python API

```python
from src.models.universal_assistant import UniversalAssistant
from src.core.meta_learner import MAMLLearner

# Initialize with custom parameters
assistant = UniversalAssistant(
    backbone="transformer",
    meta_learner="maml",
    task_domains=["nlp", "vision", "code"],
    embedding_dim=768,
    adaptation_steps=5
)

# Custom adaptation process
assistant.meta_learner.adapt(
    task_name="my_custom_task",
    support_set=my_examples,
    inner_steps=10,
    inner_lr=0.01
)
```

### Extending with New Models

Create a new model class in `src/models/` that inherits from the base classes.

### Adding New Task Types

Implement a new task handler in `src/tasks/` following the task interface.

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check internet connection for downloading pre-trained models
   - Ensure enough disk space for model storage

2. **CUDA Out of Memory**
   - Reduce batch size in configs
   - Use a smaller backbone model

3. **Adaptation Not Working**
   - Ensure examples are diverse and representative
   - Try more examples (5-10 per class)
   - Adjust adaptation steps and learning rate

### Getting Help

- Check the issues on the GitHub repository
- Search the documentation for specific terms
- Contact the project maintainers

## API Reference

### Core Classes

- `UniversalAssistant`: Main interface for the assistant
- `MAMLLearner`: Implementation of Model-Agnostic Meta-Learning
- `ProtoNetLearner`: Implementation of Prototypical Networks

### Key Methods

- `register_task(name, type, modality, **kwargs)`: Register a new task
- `adapt(task_name, input_data, expected_output)`: Adapt to a task
- `predict(task_name, input_data)`: Make predictions
- `save_model(path)`: Save the current model state
- `load_model(path)`: Load a model state

For complete API documentation, see the docstrings in the source code.
