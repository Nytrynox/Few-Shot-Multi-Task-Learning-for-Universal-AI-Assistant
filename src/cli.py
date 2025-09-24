"""
Command Line Interface for the Universal AI Assistant.

Provides a simple CLI for interacting with the Universal AI Assistant
for quick testing and demonstration purposes.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    from src.models.universal_assistant import create_universal_assistant
    from src.tasks.base import TaskType
    try:
        from src.utils.logging import configure_logging, get_logger
    except ImportError:
        # Use basic logging if custom logging module is not available
        configure_logging = lambda **kwargs: None
        get_logger = lambda name: logging.getLogger(name)
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure all dependencies are installed: pip install torch transformers higher numpy pandas matplotlib")
    sys.exit(1)


class DummyAssistant:
    """Dummy assistant for demonstration when dependencies are missing."""
    
    def __init__(self):
        self.registered_tasks = {
            "demo_sentiment": {
                "task_type": TaskType.CLASSIFICATION if 'TaskType' in globals() else "classification",
                "modality": "text",
                "num_classes": 2
            },
            "demo_qa": {
                "task_type": TaskType.GENERATION if 'TaskType' in globals() else "generation",
                "modality": "text",
                "num_classes": None
            }
        }
    
    def register_task(self, task_name, task_type, modality, num_classes=None):
        self.registered_tasks[task_name] = {
            "task_type": task_type,
            "modality": modality,
            "num_classes": num_classes
        }
        return {"status": "success"}
    
    def adapt(self, task_name, support_examples, n_shots=None):
        return {"status": "success", "final_loss": 0.1}
    
    def predict(self, inputs, task_name):
        import random
        task = self.registered_tasks.get(task_name, {})
        task_type = task.get("task_type", None)
        
        if str(task_type).endswith("CLASSIFICATION"):
            num_classes = task.get("num_classes", 2)
            return random.randint(0, num_classes-1)
        else:
            responses = [
                "I found the answer to your question.",
                "This is a demo response.",
                "The model would generate text here.",
                "Few-shot learning is powerful!"
            ]
            return random.choice(responses)
    
    def load_checkpoint(self, path):
        pass


class UniversalAssistantCLI:
    """Command line interface for Universal AI Assistant."""
    
    def __init__(self, config_path: str = None, model_path: str = None):
        try:
            configure_logging(console_level='INFO')
            self.logger = get_logger(__name__)
        except Exception:
            self.logger = logging.getLogger(__name__)
        
        # Default configuration
        default_config = {
            'backbone': {
                'text_model_name': 'microsoft/DialoGPT-medium',
                'vision_model_name': 'google/vit-base-patch16-224',
                'hidden_dim': 768,
                'output_dim': 512
            },
            'task_domains': ['nlp', 'vision', 'code'],
            'meta_algorithm': 'maml',
            'embed_dim': 512,
            'default_tasks': {
                'sentiment_analysis': {
                    'type': 'classification',
                    'modality': 'text',
                    'num_classes': 2
                },
                'text_classification': {
                    'type': 'classification',
                    'modality': 'text',
                    'num_classes': 5
                }
            }
        }
        
        # Create assistant
        self.logger.info("Initializing Universal AI Assistant...")
        try:
            self.assistant = create_universal_assistant(default_config)
            
            if model_path and Path(model_path).exists():
                try:
                    self.assistant.load_checkpoint(model_path)
                    self.logger.info(f"Loaded model from {model_path}")
                except Exception as e:
                    self.logger.warning(f"Could not load model: {e}")
            
            self.logger.info("Universal AI Assistant ready!")
        except Exception as e:
            self.logger.error(f"Failed to initialize Universal AI Assistant: {e}")
            self.logger.error("Using dummy assistant mode for demonstration purposes.")
            self.assistant = DummyAssistant()
        
    def list_tasks(self):
        """List all registered tasks."""
        print("\\nRegistered Tasks:")
        print("-" * 40)
        
        if not self.assistant.registered_tasks:
            print("No tasks registered.")
            return
        
        for task_name, task_config in self.assistant.registered_tasks.items():
            task_type = task_config['task_type'].value
            modality = task_config['modality']
            num_classes = task_config.get('num_classes', 'N/A')
            
            print(f"• {task_name}")
            print(f"  Type: {task_type}")
            print(f"  Modality: {modality}")
            print(f"  Classes: {num_classes}")
            print()
    
    def register_task(self, task_name: str, task_type: str, modality: str, num_classes: int = None):
        """Register a new task."""
        try:
            task_type_enum = TaskType(task_type.lower())
            self.assistant.register_task(
                task_name=task_name,
                task_type=task_type_enum,
                modality=modality,
                num_classes=num_classes
            )
            print(f"✅ Successfully registered task: {task_name}")
        except Exception as e:
            print(f"❌ Error registering task: {e}")
    
    def adapt_task(self, task_name: str, examples: list):
        """Adapt to a task using provided examples."""
        if task_name not in self.assistant.registered_tasks:
            print(f"❌ Task '{task_name}' not found. Use 'list' to see registered tasks.")
            return
        
        try:
            # Convert examples to proper format
            support_examples = []
            for example in examples:
                if isinstance(example, dict):
                    support_examples.append(example)
                else:
                    # Simple text example
                    support_examples.append({
                        'inputs': {'text': str(example)},
                        'target': 0  # Default target
                    })
            
            result = self.assistant.adapt(
                task_name=task_name,
                support_examples=support_examples,
                n_shots=len(support_examples)
            )
            
            print(f"✅ Successfully adapted to task: {task_name}")
            print(f"Adaptation loss: {result.get('final_loss', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error during adaptation: {e}")
    
    def predict(self, task_name: str, input_text: str):
        """Make a prediction on input text."""
        if task_name not in self.assistant.registered_tasks:
            print(f"❌ Task '{task_name}' not found. Use 'list' to see registered tasks.")
            return
        
        try:
            inputs = {'text': input_text}
            prediction = self.assistant.predict(inputs, task_name)
            
            task_config = self.assistant.registered_tasks[task_name]
            task_type = task_config['task_type']
            
            if task_type == TaskType.CLASSIFICATION:
                if hasattr(prediction, 'shape') and len(prediction.shape) > 0:
                    pred_class = prediction.argmax().item()
                    confidence = prediction.softmax(dim=-1).max().item()
                    print(f"Prediction: Class {pred_class} (confidence: {confidence:.3f})")
                else:
                    print(f"Prediction: {prediction}")
            else:
                print(f"Prediction: {prediction}")
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
    
    def interactive_mode(self):
        """Start interactive mode."""
        print("\\n🚀 Universal AI Assistant - Interactive Mode")
        print("=" * 50)
        print("Commands:")
        print("  list                    - List registered tasks")
        print("  register <name> <type> <modality> [classes] - Register new task")
        print("  adapt <task> <text>     - Adapt to task with example")
        print("  predict <task> <text>   - Make prediction")
        print("  help                    - Show this help")
        print("  quit                    - Exit interactive mode")
        print("=" * 50)
        
        while True:
            try:
                command = input("\\n> ").strip()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                
                elif cmd == 'help':
                    print("Available commands:")
                    print("  list - List all registered tasks")
                    print("  register <name> <type> <modality> [classes] - Register a new task")
                    print("  adapt <task> <text> - Adapt to task using text example")
                    print("  predict <task> <text> - Make prediction on text")
                    print("  quit - Exit")
                
                elif cmd == 'list':
                    self.list_tasks()
                
                elif cmd == 'register':
                    if len(parts) < 4:
                        print("Usage: register <name> <type> <modality> [num_classes]")
                        continue
                    
                    task_name = parts[1]
                    task_type = parts[2]
                    modality = parts[3]
                    num_classes = int(parts[4]) if len(parts) > 4 else None
                    
                    self.register_task(task_name, task_type, modality, num_classes)
                
                elif cmd == 'adapt':
                    if len(parts) < 3:
                        print("Usage: adapt <task> <text>")
                        continue
                    
                    task_name = parts[1]
                    text = ' '.join(parts[2:])
                    
                    self.adapt_task(task_name, [text])
                
                elif cmd == 'predict':
                    if len(parts) < 3:
                        print("Usage: predict <task> <text>")
                        continue
                    
                    task_name = parts[1]
                    text = ' '.join(parts[2:])
                    
                    self.predict(task_name, text)
                
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\\n\\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Universal AI Assistant CLI")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive mode'
    )
    parser.add_argument(
        '--task',
        type=str,
        help='Task name for direct prediction'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input text for direct prediction'
    )
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = UniversalAssistantCLI(
        config_path=args.config,
        model_path=args.model
    )
    
    if args.interactive:
        cli.interactive_mode()
    elif args.task and args.input:
        cli.predict(args.task, args.input)
    else:
        print("Universal AI Assistant CLI")
        print("Use --interactive for interactive mode")
        print("Or use --task <name> --input <text> for direct prediction")
        cli.list_tasks()


if __name__ == '__main__':
    main()
