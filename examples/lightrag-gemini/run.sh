#!/bin/bash

# Helper script to run the Federated LightRAG example

set -e

echo "Federated LightRAG with Gemini via VertexAI"
echo "==========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your Google Cloud project details."
    echo "Then run this script again."
    exit 1
fi

# Load environment variables
source .env

# Check if Google Cloud project is configured
if [ "$GOOGLE_CLOUD_PROJECT" = "your-project-id" ]; then
    echo "Error: Please configure GOOGLE_CLOUD_PROJECT in .env file"
    exit 1
fi

echo "Using Google Cloud Project: $GOOGLE_CLOUD_PROJECT"
echo "Using Location: $GOOGLE_CLOUD_LOCATION"
echo "Using LLM Model: $LLM_MODEL"

# Check if authenticated
echo ""
echo "Checking Google Cloud authentication..."
if ! gcloud auth application-default print-access-token &>/dev/null; then
    echo "Not authenticated. Please run:"
    echo "  gcloud auth application-default login"
    exit 1
fi
echo "✓ Authenticated"

# Clean up previous run data
echo ""
echo "Cleaning up previous run data..."
rm -rf lightrag_data/

# Run the Flower simulation
echo ""
echo "Starting Federated LightRAG simulation..."
echo ""
uv run flwr run .

echo ""
echo "Simulation complete!"
