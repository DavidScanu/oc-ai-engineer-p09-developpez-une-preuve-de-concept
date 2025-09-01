import streamlit as st
import pandas as pd
import numpy as np
from utils.data_analysis import get_basic_stats, load_sentiment140_data
from utils.visualizations import create_accessible_colors
import os

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Analyse de Sentiment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS pour l'accessibilité WCAG
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stAlert {
        border-radius: 8px;
        border-width: 2px;
    }
    
    .dataset-info {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .model-selector {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_dataset_preview():
    """Obtient un aperçu du dataset pour la page d'accueil"""
    try:
        df = load_sentiment140_data()
        if df is not None and len(df) > 0:
            return {
                'loaded': True,
                'sample_size': len(df),
                'positive_count': (df['target'] == 1).sum(),
                'negative_count': (df['target'] == 0).sum(),
                'avg_length': df['text'].str.len().mean(),
                'sample_tweets': {
                    'positive': df[df['target'] == 1]['text'].head(2).tolist(),
                    'negative': df[df['target'] == 0]['text'].head(2).tolist()
                }
            }
    except Exception as e:
        st.error(f"Erreur lors du chargement du dataset: {e}")
        return {'loaded': False}

def main():
    """Page d'accueil du dashboard"""
    
    # En-tête avec titre et description
    st.title("🎯 Dashboard d'Analyse de Sentiment")
    st.markdown("""
    ---
    
    **Bienvenue dans le dashboard d'analyse de sentiment basé sur ModernBERT.**
    
    Ce dashboard présente une **preuve de concept complète** pour la détection automatique de sentiment 
    dans les textes, utilisant un **modèle ModernBERT** fine-tuné sur le **dataset Sentiment140**.
    """)
    
    # Navigation claire
    st.markdown("---")
    st.subheader("🧭 Navigation du Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Analyse Exploratoire
        **Analyse du vrai dataset Sentiment140 :**
        - Distribution des sentiments réels
        - Statistiques textuelles authentiques
        - WordCloud basé sur les vraies données
        - Analyse de fréquence des mots
        - Insights sur la longueur des tweets
        """)
        
    with col2:
        st.markdown("""
        ### 🤖 Prédiction en Temps Réel
        **Interface de prédiction avec ModernBERT :**
        - Saisie libre ou exemples prédéfinis
        - Prédiction instantanée
        - Scores de confiance détaillés
        - Historique des prédictions
        - Visualisation des probabilités
        """)
        
    with col3:
        st.markdown("""
        ### 📈 Métriques Détaillées
        **Évaluation complète du modèle :**
        - Matrice de confusion
        - Courbe ROC interactive
        - Comparaison avec baseline DistilBERT
        - Historique d'entraînement
        - Analyse d'erreurs
        """)
    
    # Informations sur l'accessibilité
    st.markdown("---")
    st.subheader("♿ Informations d'Accessibilité WCAG")
    st.markdown("""
    **Ce dashboard respecte les critères WCAG essentiels :**
    
    - ✅ **Contraste élevé** : Tous les textes respectent un ratio de contraste ≥ 4.5:1
    - ✅ **Navigation au clavier** : Tous les éléments sont accessibles via le clavier
    - ✅ **Textes alternatifs** : Les graphiques incluent des descriptions textuelles
    - ✅ **Couleurs accessibles** : Palettes adaptées aux daltoniens (rouge/vert évité)
    - ✅ **Structure sémantique** : Utilisation correcte des en-têtes et landmarks
    - ✅ **Descriptions contextuelles** : Tooltips et help text pour clarifier les métriques
    - ✅ **Taille de police** : Respect des tailles minimales recommandées
    - ✅ **Zones cliquables** : Taille minimale de 44x44 pixels respectée
    """)


    # Informations sur le dataset
    dataset_preview = get_dataset_preview()

    # Sélection du modèle à afficher
    selected_model = None
    model_manager_available = False
    
    try:
        from utils.model_manager import ModelManager
        
        manager = ModelManager()
        available_models = manager.discover_models()
        model_manager_available = True

    except ImportError:
        st.warning("⚠️ Gestionnaire de modèles non disponible")
    
    # Statut du système
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if dataset_preview['loaded']:
            st.success("✅ Dataset Sentiment140 chargé")
        else:
            st.warning("⚠️ Dataset en cours de chargement")
    
    with col2:
        # Vérifier si au moins un modèle existe
        if model_manager_available and available_models:
            st.success("✅ Modèle(s) ModernBERT disponible(s)")
        else:
            st.error("❌ Aucun modèle ModernBERT trouvé")
    
    with col3:
        st.info("🚀 Dashboard opérationnel")
    
    # Pied de page avec informations techniques
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>🚀 <strong>Projet 9 – Développez une preuve de concept :</strong> Amélioration d'un modèle d'analyse de sentiment de tweets</p>
        <p>🤖 ModernBERT-base | 📊 Sentiment140 Dataset | 🚀 Streamlit Dashboard</p>
        <p>🎓 OpenClassrooms • Parcours <a href="https://openclassrooms.com/fr/paths/795-ai-engineer" target="_blank" style="color: #1f77b4; text-decoration: none;">AI Engineer</a> | 👋 <em>Étudiant</em> : <a href="https://www.linkedin.com/in/davidscanu14/" target="_blank" style="color: #1f77b4; text-decoration: none;"><strong>David Scanu</strong></a></p>
        <p><em>Dernière mise à jour : Août 2025</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()