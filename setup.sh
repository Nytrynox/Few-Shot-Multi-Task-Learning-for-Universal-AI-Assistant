#!/bin/bash

# Setup script for Universal AI Assistant
set -e

echo "🚀 Setting up Universal AI Assistant..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1-2)
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt
# Install essential packages directly to ensure they're available
pip install torch transformers higher numpy pandas matplotlib

# Install package in development mode
echo "📦 Installing Universal AI Assistant package..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed logs checkpoints

# Download sample data (if needed)
echo "📊 Setting up sample data..."
python -c "
import os
import json

# Create sample tasks data
sample_data = {
    'sentiment_analysis': [
        {'text': 'I love this movie!', 'label': 1},
        {'text': 'This movie is terrible.', 'label': 0},
        {'text': 'Great acting and storyline.', 'label': 1},
        {'text': 'Boring and predictable.', 'label': 0},
        {'text': 'Amazing cinematography!', 'label': 1}
    ],
    'text_classification': [
        {'text': 'The weather is sunny today.', 'label': 0},
        {'text': 'Breaking news: Market closes higher.', 'label': 1},
        {'text': 'New smartphone released by tech company.', 'label': 2},
        {'text': 'Football match ends in a draw.', 'label': 3},
        {'text': 'Scientists discover new species.', 'label': 4}
    ]
}

os.makedirs('data/processed', exist_ok=True)
with open('data/processed/sample_tasks.json', 'w') as f:
    json.dump(sample_data, f, indent=2)

print('✅ Sample data created')
"

echo "🧪 Running basic tests..."
python -c "
import sys
import os
sys.path.append('.')

try:
    from src.models.universal_assistant import create_universal_assistant
    from src.core.maml import MAMLLearner
    from src.core.prototypical import PrototypicalNetworks
    print('✅ Core imports successful')
    
    # Test basic initialization
    config = {
        'backbone': {
            'text_model_name': 'microsoft/DialoGPT-medium',
            'vision_model_name': 'google/vit-base-patch16-224',
            'hidden_dim': 768,
            'output_dim': 512
        },
        'task_domains': ['nlp'],
        'meta_algorithm': 'maml',
        'embed_dim': 512
    }
    
    assistant = create_universal_assistant(config)
    print('✅ Universal Assistant initialization successful')
    
except Exception as e:
    print(f'❌ Test failed: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Quick Start:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Interactive mode: python src/cli.py --interactive"
echo "  3. Train model: python scripts/train.py --config configs/universal_assistant.yaml"
echo "  4. Run demo: jupyter notebook notebooks/universal_assistant_demo.ipynb"
echo ""
echo "For more details, see README.md"
