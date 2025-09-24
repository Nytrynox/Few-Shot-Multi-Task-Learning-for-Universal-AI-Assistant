#!/bin/bash

# Run the Universal AI Assistant GUI
# This script launches the Streamlit GUI interface

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but could not be found."
    exit 1
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 Streamlit not found. Installing..."
    pip install streamlit
fi

# Check if the GUI file exists
if [ ! -f "src/gui.py" ]; then
    echo "❌ Could not find src/gui.py"
    exit 1
fi

echo "🚀 Launching Universal AI Assistant GUI..."
echo "ℹ️  The web interface will open in your browser automatically."

# Run the Streamlit app
streamlit run src/gui.py "$@"
