# Universal AI Assistant - Quick Reference Guide

## Quick Start

### Run the GUI
```bash
chmod +x run_gui.sh
./run_gui.sh
```
Opens in browser at http://localhost:8501

### Run the CLI
```bash
python src/cli.py --interactive
```

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `list` | Show all tasks | `> list` |
| `register <name> <type> <modality> [classes]` | Create new task | `> register sentiment classification text 2` |
| `adapt <task> <text> <label>` | Add example | `> adapt sentiment "Great product!" positive` |
| `predict <task> <text>` | Make prediction | `> predict sentiment "Terrible experience"` |
| `help` | Show help | `> help` |
| `quit` | Exit CLI | `> quit` |

## GUI Navigation

- **Home**: Overview and system status
- **Task Demo**: Interactive few-shot learning
- **Task Explorer**: Create and manage tasks
- **Model Analysis**: Performance visualizations
- **Settings**: Configure parameters

## Task Types

- **Classification**: Discrete categories (sentiment, topics)
- **Generation**: Create text or code
- **Regression**: Predict numerical values

## Example Workflow

1. **Register task**: Create a new task definition
2. **Provide examples**: Add 3-5 examples per class
3. **Test adaptation**: Try new inputs to see predictions
4. **Analyze performance**: Check confidence scores and metrics

## Supported Modalities

- **Text**: Natural language, documents
- **Vision**: Images, charts, diagrams
- **Code**: Programming source code
- **Math**: Equations, proofs, problems

## Training Custom Models

```bash
# Meta-training
python scripts/meta_train.py --config configs/meta_training.yaml

# Evaluation
python scripts/evaluate.py --model_path checkpoints/best_model.pt
```

## Where to Find More Information

- Full documentation: `docs/user_guide.md`
- API reference: Source code docstrings
- Configuration options: `configs/` directory
- Example tasks: `src/static/example_tasks.json`
