FROM python:3.11-slim-bookworm

# Métadonnées
LABEL maintainer="votre-email@example.com"
LABEL description="Dashboard ModernBERT Sentiment Analysis"

# Variables d'environnement pour optimiser Python ET corriger Streamlit
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    MPLCONFIGDIR=/tmp/matplotlib

# Créer un utilisateur non-root pour la sécurité
RUN groupadd -r streamlit && useradd -r -g streamlit streamlit

# Installation des dépendances système optimisées
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    git-lfs \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances en premier (pour le cache Docker)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY --chown=streamlit:streamlit . .

# Configuration des répertoires pour matplotlib et streamlit
RUN mkdir -p /tmp/matplotlib /home/streamlit/.config && \
    chown -R streamlit:streamlit /tmp/matplotlib /home/streamlit

# Configurer Git LFS et télécharger les modèles
RUN git lfs install && \
    git lfs pull || echo "LFS files may not be available, continuing..."

# Changer vers l'utilisateur non-root
USER streamlit

# Port d'exposition (Cloud Run utilise la variable PORT)
EXPOSE 8080

# Healthcheck pour Cloud Run
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/_stcore/health || exit 1

# Commande de démarrage CORRIGÉE pour Cloud Run
CMD streamlit run main.py \
    --server.port=${PORT:-8080} \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false \
    --server.headless=true \
    --server.enableStaticServing=true