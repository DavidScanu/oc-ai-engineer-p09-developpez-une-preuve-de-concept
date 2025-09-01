#!/bin/bash

# Configuration
MODEL_PATH="models/modernbert-sentiment-20250816_1156/model/model.safetensors"
MIN_SIZE=1000000  # 1MB minimum pour considérer que ce n'est pas un pointer

echo "🔍 Checking AI model..."

# Fonction pour vérifier la taille du fichier
check_model_size() {
    if [ -f "$MODEL_PATH" ]; then
        SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt "$MIN_SIZE" ]; then
            echo "✅ Model ready ($(du -h "$MODEL_PATH" | cut -f1))"
            return 0
        else
            echo "⚠️  Model is pointer file ($(du -h "$MODEL_PATH" | cut -f1))"
            return 1
        fi
    else
        echo "❌ Model file not found"
        return 1
    fi
}

# Vérifier et télécharger si nécessaire
if ! check_model_size; then
    echo "📥 Downloading LFS files..."
    if git lfs pull; then
        echo "✅ LFS files downloaded successfully"
        check_model_size
    else
        echo "⚠️  LFS download failed, continuing with fallback..."
    fi
fi

# Démarrer Streamlit depuis le bon répertoire
echo "🚀 Starting Streamlit..."

# Lancer Streamlit avec gestion d'erreur
if command -v streamlit >/dev/null 2>&1; then
    echo "🌐 Streamlit will be available at http://localhost:8501"
    streamlit run main.py \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        --server.headless true \
        --server.address 0.0.0.0 \
        --server.port 8501
else
    echo "❌ Streamlit not found. Installing..."
    pip3 install --user streamlit
    streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false
fi