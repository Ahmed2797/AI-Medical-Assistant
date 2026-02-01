#!/bin/bash

# ================================
# Create Project Structure
# ================================

# Root files
touch app.py
touch .env
touch main.py
touch README.md
touch requirements.txt
touch notex.txt

# Project package
mkdir Data
mkdir fronted
mkdir research
mkdir -p medicalai/{chatmodel,pipeline,prompt}

# __init__.py files
touch medicalai/__init__.py
touch medicalai/chatmodel/__init__.py
touch medicalai/pipeline/__init__.py
touch medicalai/prompt/__init__.py

touch research/ai_medical_chatbot.ipynb

echo "✅ Project structure created successfully"

# bash template.sh
