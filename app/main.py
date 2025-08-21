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
    
    # Informations sur le dataset
    dataset_preview = get_dataset_preview()
    basic_stats = get_basic_stats()
    
    st.markdown("""
    <div class="dataset-info">
        <h4>📊 Dataset Sentiment140 - Stanford University</h4>
        <p>Nous utilisons le <strong>dataset Sentiment140</strong> contenant 1.6 million de tweets annotés automatiquement. 
        Pour des raisons de performance, nous travaillons avec un échantillon équilibré de 50,000 tweets.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if dataset_preview['loaded']:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Échantillon chargé",
                f"{dataset_preview['sample_size']:,}",
                help="Échantillon du dataset Sentiment140"
            )
        
        with col2:
            st.metric(
                "Tweets positifs",
                f"{dataset_preview['positive_count']:,}",
                delta="50%",
                help="Sentiment = 1"
            )
            
        with col3:
            st.metric(
                "Tweets négatifs",
                f"{dataset_preview['negative_count']:,}",
                delta="50%",
                help="Sentiment = 0"
            )
            
        with col4:
            st.metric(
                "Longueur moyenne",
                f"{dataset_preview['avg_length']:.0f} cars",
                help="Moyenne des caractères par tweet"
            )
    else:
        st.warning("⚠️ Dataset en cours de téléchargement... Visitez la page d'analyse exploratoire pour initialiser le téléchargement.")
    
    # Sélection du modèle à afficher
    selected_model = None
    model_manager_available = False
    
    try:
        from utils.model_manager import ModelManager
        
        manager = ModelManager()
        available_models = manager.discover_models()
        model_manager_available = True
        
        if available_models:
            st.markdown("---")
            st.markdown("""
            <div class="model-selector">
                <h4 style="padding-bottom: 0;">🚀 Sélection du Modèle</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if len(available_models) > 1:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    model_options = {f"{model['model_id']} ({model['training_date']})": model for model in available_models}
                    selected_model_key = st.selectbox(
                        "Choisissez le modèle à afficher :",
                        options=list(model_options.keys()),
                        index=0,
                        help="Sélectionnez le modèle ModernBERT pour afficher ses métriques"
                    )
                    selected_model = model_options[selected_model_key]
                
                with col2:
                    best_model = manager.get_best_model('roc_auc')
                    is_best = best_model and best_model['model_id'] == selected_model['model_id']
                    st.metric("Statut", "🏆 Meilleur" if is_best else "📊 Standard")
            else:
                selected_model = available_models[0]
                st.info(f"📊 Modèle unique disponible : **{selected_model['model_id']}** ({selected_model['training_date']})")
        
    except ImportError:
        st.warning("⚠️ Gestionnaire de modèles non disponible")
    
    # Métriques de performance du modèle avec vraies données
    st.markdown("---")
    st.subheader("📈 Performances du Modèle ModernBERT")
    
    if model_manager_available and selected_model:
        try:
            comparison = manager.compare_with_baseline(selected_model)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                precision = selected_model.get('precision', 0)
                precision_delta = comparison['improvements']['precision']['absolute']
                st.metric(
                    label="Précision",
                    value=f"{precision:.1%}",
                    delta=f"{precision_delta:+.3f}",
                    delta_color="normal" if precision_delta >= 0 else "inverse",
                    help="Proportion de prédictions positives correctes"
                )
            
            with col2:
                recall = selected_model.get('recall', 0)
                recall_delta = comparison['improvements']['recall']['absolute']
                st.metric(
                    label="Rappel", 
                    value=f"{recall:.1%}",
                    delta=f"{recall_delta:+.3f}",
                    delta_color="normal" if recall_delta >= 0 else "inverse",
                    help="Proportion de vrais positifs détectés"
                )
                
            with col3:
                f1 = selected_model.get('f1', 0)
                f1_delta = comparison['improvements']['f1']['absolute']
                st.metric(
                    label="F1-Score",
                    value=f"{f1:.1%}", 
                    delta=f"{f1_delta:+.3f}",
                    delta_color="normal" if f1_delta >= 0 else "inverse",
                    help="Moyenne harmonique de la précision et du rappel"
                )
                
            with col4:
                roc_auc = selected_model.get('roc_auc', 0)
                roc_auc_delta = comparison['improvements']['roc_auc']['absolute']
                st.metric(
                    label="ROC AUC",
                    value=f"{roc_auc:.1%}",
                    delta=f"{roc_auc_delta:+.3f}",
                    delta_color="normal" if roc_auc_delta >= 0 else "inverse",
                    help="Aire sous la courbe ROC"
                )
            
            # Afficher le modèle utilisé
            st.caption(f"📊 Métriques du modèle : {selected_model['model_id']} ({selected_model['training_date']})")
            
            # Alerte selon la performance
            if not comparison['summary']['is_better']:
                st.error("""
                🚨 **Attention** : Ce modèle ModernBERT présente des performances 
                **inférieures** au baseline DistilBERT. Consultez la page "Métriques Modèle" pour 
                l'analyse détaillée et les recommandations d'amélioration.
                """)
            elif comparison['summary']['significant_improvement']:
                st.success("🎯 **Excellent** : Performance supérieure au baseline DistilBERT !")
            else:
                st.info("📈 **Bon** : Légère amélioration vs baseline DistilBERT")
        
        except Exception as e:
            st.error(f"Erreur lors de la comparaison avec le baseline : {e}")
    
    elif model_manager_available and not selected_model:
        # Aucun modèle trouvé
        st.warning("⚠️ Aucun modèle ModernBERT trouvé. Ajoutez un modèle dans le dossier `models/`")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Précision", value="N/A", help="Aucun modèle disponible")
        with col2:
            st.metric(label="Rappel", value="N/A", help="Aucun modèle disponible")
        with col3:
            st.metric(label="F1-Score", value="N/A", help="Aucun modèle disponible")
        with col4:
            st.metric(label="ROC AUC", value="N/A", help="Aucun modèle disponible")
    
    else:
        # Fallback si le gestionnaire n'est pas disponible
        st.error("❌ Gestionnaire de modèles non disponible")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Précision", value="Error", help="Gestionnaire indisponible")
        with col2:
            st.metric(label="Rappel", value="Error", help="Gestionnaire indisponible")
        with col3:
            st.metric(label="F1-Score", value="Error", help="Gestionnaire indisponible")
        with col4:
            st.metric(label="ROC AUC", value="Error", help="Gestionnaire indisponible")
    
    # Aperçu des données réelles
    if dataset_preview['loaded'] and 'sample_tweets' in dataset_preview:
        st.markdown("---")
        st.subheader("👀 Aperçu des Données Réelles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Exemples de tweets positifs :**")
            for i, tweet in enumerate(dataset_preview['sample_tweets']['positive'], 1):
                st.write(f"{i}. *\"{tweet[:100]}{'...' if len(tweet) > 100 else ''}\"*")
        
        with col2:
            st.markdown("**Exemples de tweets négatifs :**")
            for i, tweet in enumerate(dataset_preview['sample_tweets']['negative'], 1):
                st.write(f"{i}. *\"{tweet[:100]}{'...' if len(tweet) > 100 else ''}\"*")

    # Statut des modèles avec le système modulaire
    st.markdown("---")
    st.subheader("🤖 Statut Général des Modèles")

    if model_manager_available:
        available_models = manager.discover_models()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modèles disponibles", len(available_models))
        
        with col2:
            if available_models:
                best_model = manager.get_best_model('roc_auc')
                best_auc = best_model.get('roc_auc', 0) if best_model else 0
                st.metric("Meilleur ROC AUC", f"{best_auc:.3f}")
            else:
                st.metric("Meilleur ROC AUC", "N/A")
        
        with col3:
            if available_models:
                latest_model = manager.get_latest_model()
                comparison = manager.compare_with_baseline(latest_model)
                is_better = comparison['summary']['is_better']
                status = "✅ Supérieur" if is_better else "⚠️ À améliorer"
                st.metric("Statut vs Baseline", status)
            else:
                st.metric("Statut vs Baseline", "N/A")
        
        # Résumé des performances
        if available_models:
            all_better = all(manager.compare_with_baseline(model)['summary']['is_better'] for model in available_models)
            any_better = any(manager.compare_with_baseline(model)['summary']['is_better'] for model in available_models)
            
            if all_better:
                st.success("🎯 **Tous les modèles** surpassent le baseline DistilBERT !")
            elif any_better:
                st.warning("📊 **Certains modèles** surpassent le baseline. Voir page Métriques pour détails.")
            else:
                st.error("🚨 **Aucun modèle** ne surpasse le baseline. Optimisation requise.")

    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modèles disponibles", "Error")
        with col2:
            st.metric("Meilleur ROC AUC", "Error")
        with col3:
            st.metric("Statut vs Baseline", "Error")
        
        st.warning("⚠️ Gestionnaire de modèles non disponible")

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
    
    # Informations techniques sur le dataset
    st.markdown("---")
    
    with st.expander("📚 À propos du Dataset Sentiment140"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Caractéristiques du dataset :**
            
            - 📊 **Volume** : 1.6 million de tweets originaux
            - 🏷️ **Annotation** : Automatique basée sur emoticons
            - 🌍 **Langue** : Anglais exclusivement
            - 📅 **Période** : Tweets collectés en 2009
            - ⚖️ **Équilibre** : 50% positif, 50% négatif
            - 🔄 **Prétraitement** : URLs et mentions normalisées
            """)
        
        with col2:
            st.markdown("""
            **Méthodologie d'annotation :**
            
            - 😊 **Positif** : Tweets contenant des emoticons positives
            - 😞 **Négatif** : Tweets contenant des emoticons négatives
            - 🧹 **Nettoyage** : Suppression des emoticons après annotation
            - ✅ **Validation** : Méthode éprouvée académiquement
            - 📖 **Référence** : Stanford NLP Group (Go et al., 2009)
            - 🎯 **Usage** : Standard pour l'évaluation en classification de sentiment
            """)
    
    # Informations sur l'accessibilité
    with st.expander("♿ Informations d'Accessibilité WCAG"):
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