#!/bin/bash

# Configuration
MODEL_PATTERN="models/*/model/model.safetensors"
MIN_SIZE=1000000

echo "🔍 Checking AI models..."

# Vérifier les modèles
check_models() {
    local found_valid=false
    
    for model_path in $MODEL_PATTERN; do
        if [ -f "$model_path" ]; then
            SIZE=$(stat -c%s "$model_path" 2>/dev/null || echo 0)
            if [ "$SIZE" -gt "$MIN_SIZE" ]; then
                echo "✅ Model ready: $model_path ($(du -h "$model_path" | cut -f1))"
                found_valid=true
            else
                echo "⚠️  Pointer file: $model_path ($(du -h "$model_path" | cut -f1))"
            fi
        fi
    done
    
    if [ "$found_valid" = true ]; then
        return 0
    else
        echo "❌ No valid models found"
        return 1
    fi
}

# Télécharger si nécessaire
if ! check_models; then
    echo "📥 Downloading LFS files..."
    if git lfs pull; then
        echo "✅ LFS files downloaded, re-checking..."
        check_models
    else
        echo "⚠️  LFS download failed, continuing anyway..."
    fi
fi

# Démarrer Streamlit
echo "🚀 Starting Streamlit..."
echo "🌐 Available at http://localhost:8501"

if command -v streamlit >/dev/null 2>&1; then
    streamlit run main.py \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        --server.headless true \
        --server.address 0.0.0.0 \
        --server.port 8501
else
    echo "❌ Installing Streamlit..."
    pip3 install --user streamlit
    streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false
fi