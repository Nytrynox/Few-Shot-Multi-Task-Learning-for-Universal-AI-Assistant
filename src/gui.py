"""
GUI application for the Universal AI Assistant using Streamlit.

This provides a web-based interface for interacting with the Universal AI
Assistant for demonstration and easy experimentation.
"""

import os
import sys
import logging
import streamlit as st
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        configure_logging(console_level='INFO')
        logger = get_logger(__name__)
    except ImportError:
        # Continue with basic logging if custom logging module is not available
        pass
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Make sure all dependencies are installed: `pip install torch transformers higher numpy pandas matplotlib streamlit`")
    st.stop()

# Create a dummy assistant if dependencies are missing
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

def initialize_assistant():
    """Initialize the Universal AI Assistant."""
    try:
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
        logger.info("Initializing Universal AI Assistant...")
        assistant = create_universal_assistant(default_config)
        logger.info("Universal AI Assistant ready!")
        return assistant
    
    except Exception as e:
        logger.error(f"Failed to initialize Universal AI Assistant: {e}")
        logger.error("Using dummy assistant mode for demonstration purposes.")
        return DummyAssistant()

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Universal AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 42px;
            font-weight: bold;
            color: #1E88E5;
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 24px;
            color: #424242;
            margin-bottom: 20px;
        }
        .task-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        .prediction-box {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">Universal AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Few-Shot Multi-Task Learning</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        with st.spinner('Initializing Universal AI Assistant...'):
            st.session_state.assistant = initialize_assistant()
            st.session_state.support_examples = {}
            st.session_state.task_results = {}
    
    # Sidebar
    with st.sidebar:
        st.header("Task Management")
        
        # Task registration
        st.subheader("Register New Task")
        task_name = st.text_input("Task Name", key="register_task_name")
        task_type = st.selectbox(
            "Task Type", 
            ["classification", "generation", "regression"], 
            key="register_task_type"
        )
        modality = st.selectbox(
            "Modality", 
            ["text", "vision", "code"], 
            key="register_modality"
        )
        
        # Only show num_classes for classification tasks
        num_classes = None
        if task_type == "classification":
            num_classes = st.number_input("Number of Classes", min_value=2, value=2, key="register_num_classes")
        
        if st.button("Register Task"):
            if task_name:
                task_type_enum = getattr(TaskType, task_type.upper()) if hasattr(TaskType, task_type.upper()) else task_type
                try:
                    st.session_state.assistant.register_task(
                        task_name=task_name,
                        task_type=task_type_enum,
                        modality=modality,
                        num_classes=num_classes
                    )
                    st.success(f"Task '{task_name}' registered successfully!")
                    # Initialize support examples for this task
                    st.session_state.support_examples[task_name] = []
                except Exception as e:
                    st.error(f"Error registering task: {e}")
            else:
                st.warning("Please enter a task name")
        
        # Display registered tasks
        st.subheader("Registered Tasks")
        registered_tasks = getattr(st.session_state.assistant, 'registered_tasks', {})
        
        if registered_tasks:
            for task_name, task_info in registered_tasks.items():
                task_type = getattr(task_info.get('task_type'), 'value', task_info.get('task_type', 'Unknown'))
                modality = task_info.get('modality', 'Unknown')
                num_classes = task_info.get('num_classes', 'N/A')
                
                st.markdown(f"**{task_name}**")
                st.markdown(f"Type: {task_type}, Modality: {modality}, Classes: {num_classes}")
                st.markdown("---")
        else:
            st.info("No tasks registered yet.")

    # Main content - Task workspace
    st.header("Task Workspace")
    
    tabs = st.tabs(["Task Adaptation", "Model Prediction"])
    
    # Task Adaptation Tab
    with tabs[0]:
        st.subheader("Task Adaptation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Select task
            task_names = list(getattr(st.session_state.assistant, 'registered_tasks', {}).keys())
            if not task_names:
                st.info("No tasks registered. Please register a task in the sidebar.")
                selected_task = None
            else:
                selected_task = st.selectbox("Select Task", task_names, key="adapt_task_select")
        
        # Add examples and adapt
        if selected_task:
            with col1:
                st.subheader(f"Add Example for '{selected_task}'")
                
                # Get task info
                task_info = st.session_state.assistant.registered_tasks.get(selected_task, {})
                modality = task_info.get('modality', 'text')
                task_type = getattr(task_info.get('task_type'), 'value', task_info.get('task_type', ''))
                num_classes = task_info.get('num_classes', 2)
                
                # Input for example
                example_text = st.text_area("Example Input", key="example_text")
                
                # Target based on task type
                target = None
                if task_type == "CLASSIFICATION" or task_type == "classification":
                    target = st.number_input("Target Class", min_value=0, max_value=num_classes-1 if num_classes else 10, value=0)
                elif task_type == "REGRESSION" or task_type == "regression":
                    target = st.number_input("Target Value", value=0.0)
                elif task_type == "GENERATION" or task_type == "generation":
                    target = st.text_area("Target Text")
                
                # Add example
                if st.button("Add Example"):
                    if example_text:
                        if selected_task not in st.session_state.support_examples:
                            st.session_state.support_examples[selected_task] = []
                            
                        example = {
                            'inputs': {'text': example_text},
                            'target': target
                        }
                        st.session_state.support_examples[selected_task].append(example)
                        st.success("Example added!")
                    else:
                        st.warning("Please enter example text")
            
            # Display examples and adapt
            with col2:
                st.subheader(f"Examples for '{selected_task}'")
                
                examples = st.session_state.support_examples.get(selected_task, [])
                
                if examples:
                    for i, example in enumerate(examples):
                        with st.expander(f"Example {i+1}", expanded=True):
                            st.write("Input:", example['inputs']['text'])
                            st.write("Target:", example['target'])
                    
                    # Adapt button
                    if st.button("Adapt to Examples", key="adapt_button"):
                        with st.spinner("Adapting to examples..."):
                            try:
                                result = st.session_state.assistant.adapt(
                                    task_name=selected_task,
                                    support_examples=examples,
                                    n_shots=len(examples)
                                )
                                st.session_state.task_results[selected_task] = result
                                st.success(f"Adapted to '{selected_task}' with {len(examples)} examples!")
                                if hasattr(result, 'get') and result.get('final_loss'):
                                    st.info(f"Final loss: {result.get('final_loss'):.4f}")
                            except Exception as e:
                                st.error(f"Error during adaptation: {e}")
                else:
                    st.info("No examples added yet. Add examples in the left panel.")
    
    # Model Prediction Tab
    with tabs[1]:
        st.subheader("Make Predictions")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Select task
            task_names = list(getattr(st.session_state.assistant, 'registered_tasks', {}).keys())
            if not task_names:
                st.info("No tasks registered. Please register a task in the sidebar.")
                prediction_task = None
            else:
                prediction_task = st.selectbox("Select Task", task_names, key="predict_task_select")
        
        if prediction_task:
            with col1:
                # Check if task has been adapted
                if prediction_task not in st.session_state.support_examples or not st.session_state.support_examples[prediction_task]:
                    st.warning(f"No examples provided for '{prediction_task}'. Prediction may not be accurate.")
                
                # Input for prediction
                prediction_text = st.text_area("Input Text", key="prediction_text")
                
                # Make prediction
                if st.button("Make Prediction"):
                    if prediction_text:
                        with st.spinner("Predicting..."):
                            try:
                                inputs = {'text': prediction_text}
                                prediction = st.session_state.assistant.predict(inputs, prediction_task)
                                
                                with col2:
                                    st.subheader("Prediction Result")
                                    
                                    # Get task info
                                    task_info = st.session_state.assistant.registered_tasks.get(prediction_task, {})
                                    task_type = getattr(task_info.get('task_type'), 'value', task_info.get('task_type', ''))
                                    
                                    # Display prediction based on task type
                                    if task_type == "CLASSIFICATION" or task_type == "classification":
                                        st.markdown(f'<div class="prediction-box">Predicted Class: {prediction}</div>', unsafe_allow_html=True)
                                    elif task_type == "REGRESSION" or task_type == "regression":
                                        st.markdown(f'<div class="prediction-box">Predicted Value: {prediction:.4f}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="prediction-box">Prediction: {prediction}</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")
                    else:
                        st.warning("Please enter text for prediction")

if __name__ == '__main__':
    main()
