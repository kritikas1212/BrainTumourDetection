#!/bin/bash

# Brain Tumor Classification Dashboard Launcher
# This script activates the virtual environment and launches the Streamlit dashboard

echo "🧠 Brain Tumor Classification Dashboard"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Using system Python."
fi

# Set python alias to python3 if needed
if ! command -v python &> /dev/null; then
    alias python=python3
fi

# Check if model file exists
if [ ! -f "resnet_model.pth" ]; then
    echo "❌ Error: Model file 'resnet_model.pth' not found!"
    echo "Please train the model first by running: python train.py"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed!"
    echo "Installing required packages..."
    if [ "$(uname)" == "Darwin" ]; then
        pip install -r requirements-mac.txt
    else
        pip install -r requirements.txt
    fi
fi

echo ""
echo "🚀 Starting dashboard..."
echo "The dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
streamlit run app.py

