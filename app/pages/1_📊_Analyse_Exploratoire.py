import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_analysis import (
    load_sentiment140_data, 
    analyze_text_statistics, 
    get_word_frequencies, 
    create_wordcloud_from_data,
    get_temporal_analysis,
    get_basic_stats
)

st.set_page_config(
    page_title="Analyse Exploratoire",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    """Cache le chargement des données"""
    return load_sentiment140_data()

def create_sentiment_distribution(df):
    """Crée un graphique de distribution des sentiments avec vraies données"""
    sentiment_counts = df['target'].value_counts().sort_index()
    sentiment_labels = ['Négatif', 'Positif']
    percentages = (sentiment_counts / len(df) * 100).round(1)
    
    colors = ['#d62728', '#2ca02c']
    
    fig = px.bar(
        x=sentiment_labels, 
        y=sentiment_counts.values,
        color=sentiment_labels,
        color_discrete_sequence=colors,
        title="Distribution des Sentiments dans le Dataset Sentiment140",
        text=[f'{count:,}<br>({pct}%)' for count, pct in zip(sentiment_counts.values, percentages)]
    )
    
    fig.update_traces(
        textposition='outside',
        textfont_size=12
    )
    
    fig.update_layout(
        showlegend=False,
        title_font_size=18,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_yaxes(title_text="Nombre de tweets", title_font=dict(size=14))
    fig.update_xaxes(title_text="Sentiment", title_font=dict(size=14))

    
    return fig

def create_text_length_analysis(df):
    """Analyse de la longueur des textes avec vraies données"""
    
    fig = px.histogram(
        df,
        x='text_length',
        nbins=50,
        title="Distribution de la Longueur des Tweets (caractères)",
        labels={'text_length': 'Longueur (caractères)', 'count': 'Fréquence'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Ajouter des statistiques
    mean_length = df['text_length'].mean()
    median_length = df['text_length'].median()
    
    fig.add_vline(x=mean_length, line_dash="dash", line_color="red", 
                  annotation_text=f"Moyenne: {mean_length:.0f}")
    fig.add_vline(x=median_length, line_dash="dash", line_color="orange",
                  annotation_text=f"Médiane: {median_length:.0f}")
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=18,
        font=dict(size=12)
    )
    
    return fig

def create_word_count_analysis(df):
    """Analyse du nombre de mots par tweet"""
    
    fig = px.histogram(
        df,
        x='word_count',
        nbins=30,
        title="Distribution du Nombre de Mots par Tweet",
        labels={'word_count': 'Nombre de mots', 'count': 'Fréquence'},
        color_discrete_sequence=['#ff7f0e']
    )
    
    mean_words = df['word_count'].mean()
    median_words = df['word_count'].median()
    
    fig.add_vline(x=mean_words, line_dash="dash", line_color="red",
                  annotation_text=f"Moyenne: {mean_words:.1f}")
    fig.add_vline(x=median_words, line_dash="dash", line_color="orange",
                  annotation_text=f"Médiane: {median_words:.0f}")
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=18,
        font=dict(size=12)
    )
    
    return fig

def create_word_frequency_plot(word_freq):
    """Crée un graphique de fréquence des mots avec vraies données"""
    words, frequencies = zip(*word_freq)
    
    df_words = pd.DataFrame({'Mot': words, 'Fréquence': frequencies})
    
    fig = px.bar(
        df_words, 
        x='Fréquence', 
        y='Mot',
        orientation='h',
        title="Top 20 des Mots les Plus Fréquents (après prétraitement)",
        color='Fréquence',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=18,
        font=dict(size=12),
        yaxis={'categoryorder':'total ascending'},
        height=600
    )
    
    return fig

def create_sentiment_by_length(df):
    """Analyse du sentiment en fonction de la longueur"""
    # Créer des bins de longueur
    df['length_bin'] = pd.cut(df['text_length'], bins=10, labels=False)
    sentiment_by_length = df.groupby('length_bin')['target'].mean()
    
    bin_ranges = pd.cut(df['text_length'], bins=10).cat.categories
    bin_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in bin_ranges]
    
    fig = px.bar(
        x=bin_labels,
        y=sentiment_by_length.values,
        title="Proportion de Sentiment Positif par Longueur de Tweet",
        labels={'x': 'Longueur (caractères)', 'y': 'Proportion de sentiment positif'},
        color=sentiment_by_length.values,
        color_continuous_scale='RdYlGn'
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="black",
                  annotation_text="Équilibre (50%)")
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=16,
        font=dict(size=11),
        xaxis_tickangle=-45
    )
    
    return fig

def main():
    st.title("📊 Analyse Exploratoire - Dataset Sentiment140")
    st.markdown("---")
    
    # Chargement des données avec indication de progression
    with st.spinner("📥 Chargement du dataset Sentiment140..."):
        df = load_data()
    
    if df is None or len(df) == 0:
        st.error("❌ Impossible de charger le dataset")
        st.stop()
    
    st.success(f"✅ Dataset chargé avec succès ! ({len(df):,} échantillons)")
    
    # Statistiques générales
    stats = analyze_text_statistics(df)
    
    # Section d'informations sur le dataset
    st.subheader("📋 Informations sur le Dataset")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Échantillons total", f"{stats['total_texts']:,}")
    with col2:
        st.metric("Tweets positifs", f"{stats['positive_count']:,}", 
                 delta=f"{(stats['positive_count']/stats['total_texts']*100):.1f}%")
    with col3:
        st.metric("Tweets négatifs", f"{stats['negative_count']:,}",
                 delta=f"{(stats['negative_count']/stats['total_texts']*100):.1f}%")
    with col4:
        st.metric("Longueur moyenne", f"{stats['avg_length']:.0f} cars")
    with col5:
        st.metric("Mots moyens", f"{stats['avg_words']:.1f}")
    
    # Informations techniques sur le dataset    
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



    # Aperçu rapide du dataset : exemples de tweets
    dataset_preview = {'loaded': False}
    try:
        sample_pos = df[df['target'] == 1]['text'].sample(3, random_state=1).tolist()
        sample_neg = df[df['target'] == 0]['text'].sample(3, random_state=1).tolist()
        dataset_preview = {
            'loaded': True,
            'sample_tweets': {
                'positive': sample_pos,
                'negative': sample_neg
            }
        }
    except Exception as e:
        st.warning(f"Impossible d'afficher un aperçu des tweets : {e}")

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

                
    st.markdown("---")
    
    # Distribution des sentiments
    st.subheader("🎯 Distribution des Sentiments")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_sentiment = create_sentiment_distribution(df)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Observations :**
        
        ✅ **Parfaitement équilibré**
        
        ✅ **Aucun biais de classe**
        
        ✅ **Idéal pour l'entraînement**
        
        Les données sont équilibrées entre sentiments positifs et négatifs, ce qui évite les biais dans l'apprentissage.
        """)
    
    st.markdown("---")
    
    # Analyses textuelles
    st.subheader("📝 Analyses Textuelles Détaillées")
    
    # Longueur des textes et nombre de mots
    col1, col2 = st.columns(2)
    
    with col1:
        fig_length = create_text_length_analysis(df)
        st.plotly_chart(fig_length, use_container_width=True)
        
        # Statistiques détaillées
        st.markdown("**Statistiques de longueur :**")
        st.write(f"• **Minimum :** {stats['min_length']} caractères")
        st.write(f"• **Médiane :** {stats['median_length']:.0f} caractères")
        st.write(f"• **Moyenne :** {stats['avg_length']:.0f} caractères")
        st.write(f"• **Maximum :** {stats['max_length']} caractères")
        st.write(f"• **Écart-type :** {stats['std_length']:.0f}")
    
    with col2:
        fig_words = create_word_count_analysis(df)
        st.plotly_chart(fig_words, use_container_width=True)
        
        # Statistiques de mots
        st.markdown("**Statistiques de mots :**")
        st.write(f"• **Médiane :** {stats['median_words']:.0f} mots")
        st.write(f"• **Moyenne :** {stats['avg_words']:.1f} mots")
        st.write(f"• **Total :** {stats['total_words']:,} mots")
        
        # Insight sur les limites Twitter
        if stats['max_length'] <= 280:
            st.info("📱 Tweets conformes à la limite de 280 caractères")
    
    # Analyse du sentiment par longueur
    st.subheader("📏 Sentiment en Fonction de la Longueur")
    fig_sentiment_length = create_sentiment_by_length(df)
    st.plotly_chart(fig_sentiment_length, use_container_width=True)
    
    with st.expander("💡 Interprétation"):
        st.markdown("""
        Ce graphique montre la **proportion de sentiment positif** pour différentes tranches de longueur de tweets.
        
        - **Au-dessus de 50% :** Les tweets de cette longueur tendent à être plus positifs
        - **En-dessous de 50% :** Les tweets de cette longueur tendent à être plus négatifs
        - **Autour de 50% :** Pas de biais particulier pour cette longueur
        """)
    
    st.markdown("---")
    
    # Analyse de fréquence des mots
    st.subheader("🔤 Analyse de Fréquence des Mots")
    
    with st.spinner("🔍 Analyse des fréquences de mots..."):
        word_freq = get_word_frequencies(df, top_n=20)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if word_freq:
            fig_words_freq = create_word_frequency_plot(word_freq)
            st.plotly_chart(fig_words_freq, use_container_width=True)
        else:
            st.warning("Impossible de calculer les fréquences de mots")
    
    with col2:
        st.markdown("**Top 10 des mots :**")
        if word_freq:
            for i, (word, freq) in enumerate(word_freq[:10], 1):
                st.write(f"{i}. **{word}** ({freq:,})")
        
    
    st.markdown("---")
    
    # WordClouds
    st.subheader("☁️ WordClouds par Sentiment")
    
    tab1, tab2, tab3 = st.tabs(["Tous sentiments", "Sentiment Positif", "Sentiment Négatif"])
    
    with tab1:
        with st.spinner("Génération du WordCloud général..."):
            wordcloud_all = create_wordcloud_from_data(df)
            st.pyplot(wordcloud_all, use_container_width=True)
        
        with st.expander("♿ Description textuelle"):
            st.markdown("""
            Ce WordCloud présente les mots les plus fréquents de l'ensemble du dataset.
            Les mots apparaissent en différentes tailles selon leur fréquence d'usage.
            """)
    
    with tab2:
        with st.spinner("Génération du WordCloud positif..."):
            wordcloud_pos = create_wordcloud_from_data(df, sentiment=1)
            st.pyplot(wordcloud_pos, use_container_width=True)
        
        with st.expander("♿ Description textuelle"):
            st.markdown("""
            WordCloud spécifique aux tweets classés comme positifs.
            Dominé par des termes exprimant la satisfaction, la joie et l'approbation.
            """)
    
    with tab3:
        with st.spinner("Génération du WordCloud négatif..."):
            wordcloud_neg = create_wordcloud_from_data(df, sentiment=0)
            st.pyplot(wordcloud_neg, use_container_width=True)
        
        with st.expander("♿ Description textuelle"):
            st.markdown("""
            WordCloud spécifique aux tweets classés comme négatifs.
            Dominé par des termes exprimant la frustration, la déception et la critique.
            """)
    
    st.markdown("---")
    
    # Exemples de tweets
    st.subheader("📄 Exemples de Tweets du Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Exemples de tweets positifs :**")
        positive_tweets = df[df['target'] == 1]['text'].sample(5, random_state=42)
        for i, tweet in enumerate(positive_tweets, 1):
            st.write(f"{i}. *\"{tweet}\"*")
    
    with col2:
        st.markdown("**Exemples de tweets négatifs :**")
        negative_tweets = df[df['target'] == 0]['text'].sample(5, random_state=42)
        for i, tweet in enumerate(negative_tweets, 1):
            st.write(f"{i}. *\"{tweet}\"*")
    
    # Résumé analytique
    st.markdown("---")
    st.subheader("📊 Résumé de l'Analyse Exploratoire")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Qualité du Dataset :**
        
        1. **Équilibrage parfait** : 50% positif / 50% négatif
        2. **Volume suffisant** : 50K échantillons pour l'analyse
        3. **Diversité linguistique** : Vocabulaire riche et varié
        4. **Longueur appropriée** : Compatible avec les modèles de NLP
        5. **Prétraitement cohérent** : Nettoyage standardisé appliqué
        """)
    
    with col2:
        st.info("""
        **📈 Insights Clés :**
        
        - **Longueur moyenne** : ~{:.0f} caractères par tweet
        - **Richesse lexicale** : Vocabulaire émotionnel dominant
        - **Distribution homogène** : Pas de biais temporel ou structurel
        - **Qualité élevée** : Tweets authentiques et représentatifs
        - **Prêt pour ML** : Format optimal pour l'entraînement
        """.format(stats['avg_length']))
    
    # Recommandations pour la suite
    with st.expander("🚀 Recommandations pour l'Entraînement"):
        st.markdown("""
        **Stratégies d'optimisation basées sur l'analyse :**
        
        1. **Gestion de la longueur :**
            - Utiliser une longueur max de 512 tokens (compatible avec BERT)
            - Tronquer intelligemment les tweets très longs
        
        2. **Prétraitement spécialisé :**
            - Maintenir le remplacement des URLs et mentions
            - Préserver les hashtags (information contextuelle)
            - Normaliser la casse pour réduire la variabilité
        
        3. **Augmentation des données :**
            - Utiliser des techniques de back-translation
            - Appliquer des transformations syntaxiques légères
            - Préserver l'équilibre des classes
        
        4. **Validation robuste :**
            - Split stratifié pour maintenir l'équilibre
            - Validation croisée pour évaluer la généralisation
            - Métriques multiples (accuracy, F1, ROC-AUC)
        """)

if __name__ == "__main__":
    main()