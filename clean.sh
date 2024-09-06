#!/bin/bash

# Clean up script to remove unnecessary files and directories

# Remove the 'repo' directory and its contents
rm -rf repo

# Remove all Python bytecode files (*.pyc)
find . -name "*.pyc" -type f -delete

# Optional: Remove __pycache__ directories
find . -name "__pycache__" -type d -exec rm -rf {} +

# Optional: Remove .DS_Store files (macOS)
find . -name ".DS_Store" -type f -delete
