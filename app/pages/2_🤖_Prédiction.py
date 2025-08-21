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

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Prédiction de Sentiment",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Charge le modèle ModernBERT et le tokenizer"""
    model_path = "models/modernbert-sentiment-20250816_1156/model"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        return model, tokenizer, True
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None, False

def preprocess_text(text):
    """Prétraite le texte comme pendant l'entraînement"""
    if not isinstance(text, str):
        return ""
    
    text = str(text).strip()
    
    # Remplacer les URLs par un token spécial
    import re
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Remplacer les mentions par un token spécial  
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

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
    st.title("🤖 Prédiction de Sentiment en Temps Réel")
    st.markdown("---")
    
    # Chargement du modèle
    with st.spinner("Chargement du modèle ModernBERT..."):
        model, tokenizer, success = load_model()
    
    if not success:
        st.error("❌ Impossible de charger le modèle. Vérifiez que les fichiers sont présents.")
        st.stop()
    
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
                placeholder="Ex: I love this movie! It's absolutely amazing...",
                height=120,
                help="Saisissez le texte dont vous voulez analyser le sentiment"
            )
        else:
            examples = {
                "Très positif": "I absolutely love this product! It exceeded all my expectations and made my day so much better!",
                "Positif": "This is pretty good, I'm satisfied with the quality.",
                "Neutre": "The weather is okay today, nothing special.",
                "Négatif": "I don't really like this, it could be better.",
                "Très négatif": "This is absolutely terrible! Worst experience ever, completely disappointed and frustrated!"
            }
            
            selected_example = st.selectbox(
                "Choisissez un exemple :",
                list(examples.keys())
            )
            user_text = examples[selected_example]
            st.text_area("Texte sélectionné :", value=user_text, height=120, disabled=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
        
        # Détails de l'analyse
        with st.expander("🔍 Détails de l'Analyse"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Texte original :**")
                st.code(user_text)
                
                st.markdown("**Texte prétraité :**")
                st.code(result['processed_text'])
            
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
            }).applymap(
                lambda x: 'color: green' if x == 'Positif' else 'color: red' if x == 'Négatif' else '',
                subset=['sentiment']
            ),
            use_container_width=True
        )
        
        # Bouton pour vider l'historique
        if st.button("🗑️ Vider l'historique"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Informations techniques
    st.markdown("---")
    
    with st.expander("🔧 Informations Techniques"):
        st.markdown("""
        **Architecture du Modèle :**
        - **Base** : ModernBERT-base (Answer.AI)
        - **Paramètres** : 149.6M total, 1.5K entraînables (fine-tuning)
        - **Tâche** : Classification binaire de sentiment
        - **Tokenizer** : ModernBERT (50K tokens)
        
        **Prétraitement :**
        - Remplacement des URLs par `[URL]`
        - Remplacement des mentions par `[USER]`
        - Normalisation des espaces
        - Troncature à 512 tokens maximum
        
        **Performance :**
        - Accuracy : 85.2%
        - F1-Score : 84.4%
        - ROC AUC : 91.7%
        """)

if __name__ == "__main__":
    main()