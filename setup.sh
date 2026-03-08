#!/bin/bash
# ============================================================
#  Research Q&A Bot — First-Time Setup Script
#  Run this once:  bash setup.sh
# ============================================================

echo ""
echo "🤖  Research Paper Q&A Bot — Setup"
echo "===================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌  Python 3 not found. Install from https://python.org"
    exit 1
fi
echo "✅  Python $(python3 --version)"

# Create virtual environment
echo ""
echo "📦  Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥  Installing dependencies..."
pip install -q -r requirements.txt
echo "✅  Dependencies installed."

# Create papers directory
mkdir -p papers
echo "✅  Created ./papers directory."

# Prompt for API key
echo ""
echo "🔑  You need an OpenAI API key."
echo "    Get one free at: https://platform.openai.com/api-keys"
echo ""
read -p "   Paste your API key here (or press Enter to skip): " KEY

if [ ! -z "$KEY" ]; then
    echo "export OPENAI_API_KEY=\"$KEY\"" >> ~/.bashrc
    echo "export OPENAI_API_KEY=\"$KEY\"" >> ~/.zshrc 2>/dev/null
    export OPENAI_API_KEY="$KEY"
    echo "✅  API key saved to ~/.bashrc"
fi

echo ""
echo "✅  Setup complete!"
echo ""
echo "   NEXT STEPS:"
echo "   1. Add your PDF papers to ./papers/"
echo "   2. Run the terminal bot:  python app.py"
echo "   3. OR run the web UI:     streamlit run streamlit_app.py"
echo ""
