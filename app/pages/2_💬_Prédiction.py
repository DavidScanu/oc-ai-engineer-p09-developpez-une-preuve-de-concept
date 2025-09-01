import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import re
import demoji

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add model manager import
try:
    from utils.model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False

st.set_page_config(
    page_title="Prédiction de Sentiment",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model_by_path(model_path):
    """Charge un modèle spécifique par son chemin"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        return model, tokenizer, True
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None, False

@st.cache_data
def get_available_models():
    """Récupère les modèles disponibles"""
    if MODEL_MANAGER_AVAILABLE:
        manager = ModelManager()
        return manager.discover_models()
    return []


def preprocess_text(tweet):
    """
    Prétraite un tweet pour l'entraînement BERT en conservant la structure naturelle
    du langage mais en normalisant certains éléments spécifiques aux réseaux sociaux.
    """
    # Vérifier si le tweet est une chaîne de caractères
    if not isinstance(tweet, str):
        return ""

    tweet = str(tweet)
    tweet = ' '.join(tweet.split())

    # Remplacer les URLs par un token spécial
    tweet = re.sub(r'https?://\S+|www\.\S+', '[URL]', tweet)

    # Remplacer les mentions par un token spécial
    tweet = re.sub(r'@\w+', '[USER]', tweet)

    # Extraire le contenu du hashtag (supprimer le symbole #)
    tweet = re.sub(r'#(\w+)', r'\1', tweet)  # #Python -> Python

    # Convertir les émojis en descriptions textuelles
    tweet = demoji.replace_with_desc(tweet, sep=" ")

    # Normaliser les espaces multiples
    tweet = re.sub(r'\s+', ' ', tweet)

    # Nettoyer les espaces en début et fin
    tweet = tweet.strip()

    return tweet

def predict_sentiment(text, model, tokenizer):
    """Prédit le sentiment d'un texte"""
    # Prétraitement
    processed_text = preprocess_text(text)
    
    # Tokenisation
    inputs = tokenizer(
        processed_text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Conversion en format lisible
    sentiment_label = "Positif" if predicted_class == 1 else "Négatif"
    sentiment_emoji = "😊" if predicted_class == 1 else "😞"
    
    return {
        'sentiment': sentiment_label,
        'emoji': sentiment_emoji,
        'confidence': confidence,
        'probabilities': {
            'Négatif': probabilities[0][0].item(),
            'Positif': probabilities[0][1].item()
        },
        'processed_text': processed_text
    }

def create_confidence_chart(probabilities):
    """Crée un graphique des probabilités"""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ['#d62728', '#2ca02c']  # Rouge et vert accessibles
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Distribution des Probabilités",
        title_font_size=16,
        yaxis_title="Probabilité",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig



def main():
    st.title("💬 Prédiction de Sentiment en Temps Réel")
    st.markdown("---")
    
    # Sélection du modèle
    selected_model = None
    model_info = None
    
    if MODEL_MANAGER_AVAILABLE:
        available_models = get_available_models()
        
        if available_models:
            st.subheader("🔧 Sélection du Modèle")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                model_options = {f"{model['model_id']} ({model['training_date']})": model for model in available_models}
                selected_model_key = st.selectbox(
                    "Choisissez le modèle à utiliser :",
                    options=list(model_options.keys()),
                    index=0,
                    help="Sélectionnez le modèle ModernBERT pour les prédictions"
                )
                selected_model = model_options[selected_model_key]
                model_info = selected_model
            
            with col2:
                # Statut du modèle
                manager = ModelManager()
                best_model = manager.get_best_model('roc_auc')
                is_best = best_model and best_model['model_id'] == selected_model['model_id']
                st.metric("Statut", "🏆 Meilleur" if is_best else "📊 Standard")
            
            # Métriques du modèle sélectionné
            with st.expander("📊 Performances du Modèle Sélectionné"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    accuracy = selected_model.get('accuracy', 0)
                    st.metric("Accuracy", f"{accuracy:.1%}")
                with col2:
                    f1 = selected_model.get('f1', 0)
                    st.metric("F1-Score", f"{f1:.1%}")
                with col3:
                    precision = selected_model.get('precision', 0)
                    st.metric("Précision", f"{precision:.1%}")
                with col4:
                    roc_auc = selected_model.get('roc_auc', 0)
                    st.metric("ROC AUC", f"{roc_auc:.1%}")
            
            model_path = os.path.join(selected_model['model_path'], "model")
        else:
            st.error("❌ Aucun modèle ModernBERT trouvé")
            st.stop()
    else:
        st.warning("⚠️ Gestionnaire de modèles non disponible, utilisation du modèle par défaut")
        model_path = "models/modernbert-sentiment-20250816_1156/model"
    
    # Chargement du modèle
    with st.spinner("Chargement du modèle ModernBERT..."):
        model, tokenizer, success = load_model_by_path(model_path)
    
    if not success:
        st.error("❌ Impossible de charger le modèle. Vérifiez que les fichiers sont présents.")
        st.stop()
    
    if selected_model:
        st.success(f"✅ Modèle **{selected_model['model_id']}** chargé avec succès !")
    else:
        st.success("✅ Modèle ModernBERT chargé avec succès !")
    
    # Interface de prédiction
    st.subheader("💬 Interface de Prédiction")
    
    # Zone de saisie
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Options de saisie
        input_method = st.radio(
            "Méthode de saisie :",
            ["Saisie libre", "Exemples prédéfinis"],
            help="Choisissez comment entrer votre texte"
        )
        
        if input_method == "Saisie libre":
            user_text = st.text_area(
                "Entrez votre texte :",
                placeholder="Ex: 😍 I love this movie! #absolutely #amazing",
                height=120,
                help="Saisissez le texte dont vous voulez analyser le sentiment"
            )
        else:
            examples = {
                "Très positif": "OMG just got tickets for @taylorswift13 concert!! Best day EVER! 💕 #Swifties #DreamsComeTrue #SoHappy",
                "Positif": "Thanks @starbucks for the great service today! The new latte is pretty good 👍 #coffee #goodmorning",
                "Neutre": "Waiting for the bus on 5th street ⌚. Traffic looks normal today. @citybus any delays? #commute",
                "Négatif": "Ugh @netflix why did you cancel my favorite show?? Really disappointed with this decision 😒 #SaveOurShow",
                "Très négatif": "@airlinecompany WORST FLIGHT EVER! ❌ 3 hours delayed, lost luggage, rude staff! Never flying with you again!! #TravelNightmare #Angry"
            }

            selected_example = st.selectbox(
                "Choisissez un exemple :",
                list(examples.keys())
            )
            user_text = examples[selected_example]
            
            st.markdown("**Tweet d'exemple :**")
            st.markdown(f"""
            <div style="background-color: #1da1f2; color: white; padding: 12px; border-radius: 8px; margin: 10px 0;">
            {user_text}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        **Guide d'utilisation :**
        
        1. 📝 Saisissez ou sélectionnez un texte
        2. 🚀 Cliquez sur "Analyser le sentiment"
        3. 📊 Consultez les résultats et probabilités
        4. 📈 L'historique se met à jour automatiquement
        
        **Conseils :**
        - Textes en anglais pour de meilleurs résultats
        - 10 à 280 caractères recommandés
        - Les hashtags (#) sont acceptés
        - Les emojis sont acceptés
        """)
    
    # Bouton de prédiction
    predict_button = st.button("🚀 Analyser le Sentiment", type="primary")
    
    # Prédiction et affichage des résultats
    if predict_button and user_text.strip():
        with st.spinner("Analyse en cours..."):
            result = predict_sentiment(user_text, model, tokenizer)
        
        st.markdown("---")
        st.subheader("📊 Résultats de l'Analyse")
        
        # Résultat principal
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            sentiment_color = "green" if result['sentiment'] == "Positif" else "red"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; border: 2px solid {sentiment_color}; border-radius: 10px;">
                <h2 style="color: {sentiment_color}; margin: 0;">{result['emoji']} {result['sentiment']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Confiance",
                f"{result['confidence']:.1%}",
                help="Degré de certitude du modèle pour cette prédiction"
            )
        
        with col3:
            # Graphique des probabilités
            fig = create_confidence_chart(result['probabilities'])
            st.plotly_chart(fig, width='stretch')
        
        # Détails de l'analyse
        with st.expander("🔍 Détails de l'Analyse"):
            col1, col2 = st.columns(2)
            
        with col1:
            st.markdown("**Texte original :**")
            st.markdown(f"""
            <div style="background-color: #1da1f2; color: white; padding: 10px; border-radius: 5px;">
            {user_text}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Texte prétraité :**")
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 3px solid #ccc;">
            {result['processed_text']}
            </div>
            """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Probabilités détaillées :**")
                for sentiment, prob in result['probabilities'].items():
                    st.write(f"• {sentiment}: {prob:.3f} ({prob:.1%})")
                
                st.markdown("**Métadonnées :**")
                st.write(f"• Longueur: {len(user_text)} caractères")
                st.write(f"• Mots: {len(user_text.split())} mots")
                st.write(f"• Timestamp: {datetime.now().strftime('%H:%M:%S')}")
        
        # Sauvegarde dans l'historique (session state)
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        st.session_state.prediction_history.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'text': user_text[:50] + "..." if len(user_text) > 50 else user_text,
            'sentiment': result['sentiment'],
            'confidence': result['confidence']
        })
    
    elif predict_button:
        st.warning("⚠️ Veuillez saisir un texte à analyser.")
    
    # Historique des prédictions
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📈 Historique des Prédictions")
        
        # Tableau de l'historique
        df_history = pd.DataFrame(st.session_state.prediction_history)
        
        # Affichage du tableau avec formatage
        st.dataframe(
            df_history.style.format({
                'confidence': '{:.1%}'
            }).map(
                lambda x: 'color: green' if x == 'Positif' else 'color: red' if x == 'Négatif' else '',
                subset=['sentiment']
            ),
            width='stretch'
        )
        
        # Bouton pour vider l'historique
        if st.button("🗑️ Vider l'historique"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Informations techniques
    st.markdown("---")
    
    with st.expander("🔧 Informations Techniques"):
        if model_info:
            # Informations réelles du modèle
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Architecture du Modèle :**")
                st.write(f"- **Base** : {model_info.get('architecture', 'ModernBERT-base')}")
                st.write(f"- **Paramètres totaux** : {model_info.get('total_params', 0):,}")
                st.write(f"- **Paramètres entraînables** : {model_info.get('trainable_params', 0):,}")
                st.write(f"- **Vocabulaire** : {model_info.get('vocab_size', 0):,} tokens")
                st.write(f"- **Tâche** : Classification binaire de sentiment")
                
                st.markdown("**Entraînement :**")
                st.write(f"- **Dataset** : {model_info.get('total_samples', 0):,} échantillons")
                st.write(f"- **Époques** : {model_info.get('epochs_completed', 'N/A')}")
                st.write(f"- **Temps** : {model_info.get('training_time_minutes', 0):.1f} minutes")
            
            with col2:
                st.markdown("**Performances :**")
                st.write(f"- **Accuracy** : {model_info.get('accuracy', 0):.1%}")
                st.write(f"- **F1-Score** : {model_info.get('f1', 0):.1%}")
                st.write(f"- **Précision** : {model_info.get('precision', 0):.1%}")
                st.write(f"- **Rappel** : {model_info.get('recall', 0):.1%}")
                st.write(f"- **ROC AUC** : {model_info.get('roc_auc', 0):.1%}")
                st.write(f"- **Loss finale** : {model_info.get('loss', 0):.4f}")
                
                st.markdown("**Prétraitement :**")
                st.write("- Remplacement des URLs par `[URL]`")
                st.write("- Remplacement des mentions par `[USER]`")
                st.write("- Suppression des hashtags (#)")
                st.write("- Normalisation des espaces")
                st.write("- Troncature à 512 tokens maximum")
        else:
            # Informations par défaut
            st.markdown("""
            **Architecture du Modèle :**
            - **Base** : ModernBERT-base (Answer.AI)
            - **Paramètres** : 149.6M total, 1.5K entraînables (fine-tuning)
            - **Tâche** : Classification binaire de sentiment
            - **Tokenizer** : ModernBERT (50K tokens)
            
            **Prétraitement :**
            - Remplacement des URLs par `[URL]`
            - Remplacement des mentions par `[USER]`
            - Suppression des hashtags (#)
            - Remplacement des émojis par leur description textuelle
            - Normalisation des espaces
            - Troncature à 512 tokens maximum
            """)



if __name__ == "__main__":
    main()