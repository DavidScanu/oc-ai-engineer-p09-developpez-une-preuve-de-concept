import pandas as pd
import numpy as np
import requests
import zipfile
import os
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def download_sentiment140_dataset(force_download=False):
    """Télécharge et extrait le dataset Sentiment140"""
    
    url = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+7%C2%A0-+D%C3%A9tectez+les+Bad+Buzz+gr%C3%A2ce+au+Deep+Learning/sentiment140.zip"
    
    # Chemins
    data_dir = "data"
    local_zip_path = os.path.join(data_dir, "sentiment140.zip")
    csv_file_path = os.path.join(data_dir, "training.1600000.processed.noemoticon.csv")
    sample_file_path = os.path.join(data_dir, "sentiment140_sample.csv")
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs(data_dir, exist_ok=True)
    
    # Vérifier si le fichier échantillon existe déjà
    if os.path.exists(sample_file_path) and not force_download:
        print("✅ Dataset échantillon déjà présent")
        return sample_file_path
    
    try:
        # Télécharger le fichier ZIP
        print("📥 Téléchargement du dataset Sentiment140...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print("✅ Téléchargement terminé")
        
        # Extraire le fichier ZIP
        print("📂 Extraction du fichier...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Supprimer le fichier ZIP
        os.remove(local_zip_path)
        
        # Créer un échantillon pour les performances
        print("🔄 Création d'un échantillon pour l'analyse...")
        create_sample_dataset(csv_file_path, sample_file_path)
        
        # Supprimer le fichier complet pour économiser l'espace
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        
        print("✅ Dataset préparé avec succès")
        return sample_file_path
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return None

def create_sample_dataset(full_path, sample_path, sample_size=50000):
    """Crée un échantillon équilibré du dataset complet"""
    
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    
    print(f"📊 Lecture du dataset complet...")
    # Lire par chunks pour éviter les problèmes de mémoire
    chunk_size = 100000
    sample_data = []
    
    positive_count = 0
    negative_count = 0
    target_per_class = sample_size // 2
    
    for chunk in pd.read_csv(full_path, encoding='latin-1', names=column_names, chunksize=chunk_size):
        # Convertir les labels (4 -> 1 pour positif)
        chunk['target'] = chunk['target'].replace(4, 1)
        
        # Échantillonner de manière équilibrée
        pos_chunk = chunk[chunk['target'] == 1]
        neg_chunk = chunk[chunk['target'] == 0]
        
        # Ajouter des échantillons positifs
        if positive_count < target_per_class and len(pos_chunk) > 0:
            needed = min(target_per_class - positive_count, len(pos_chunk))
            sample_data.append(pos_chunk.sample(n=needed, random_state=42))
            positive_count += needed
        
        # Ajouter des échantillons négatifs
        if negative_count < target_per_class and len(neg_chunk) > 0:
            needed = min(target_per_class - negative_count, len(neg_chunk))
            sample_data.append(neg_chunk.sample(n=needed, random_state=42))
            negative_count += needed
        
        # Arrêter si on a assez d'échantillons
        if positive_count >= target_per_class and negative_count >= target_per_class:
            break
    
    # Combiner et mélanger
    if sample_data:
        df_sample = pd.concat(sample_data).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Nettoyer les textes
        df_sample['text'] = df_sample['text'].apply(clean_tweet)
        df_sample = df_sample[df_sample['text'].str.len() > 0]
        
        # Sauvegarder
        df_sample.to_csv(sample_path, index=False)
        print(f"✅ Échantillon créé: {len(df_sample)} tweets ({positive_count} positifs, {negative_count} négatifs)")
        
        return df_sample
    else:
        print("❌ Erreur lors de la création de l'échantillon")
        return None

def clean_tweet(text):
    """Nettoie un tweet"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = ' '.join(text.split())
    return text

def preprocess_tweet(tweet):
    """Prétraite un tweet comme dans l'entraînement"""
    if not isinstance(tweet, str):
        return ""
    
    tweet = str(tweet)
    tweet = ' '.join(tweet.split())
    
    # Remplacer les URLs
    tweet = re.sub(r'https?://\S+|www\.\S+', '[URL]', tweet)
    
    # Remplacer les mentions
    tweet = re.sub(r'@\w+', '[USER]', tweet)
    
    # Normaliser les hashtags
    tweet = re.sub(r'#(\w+)', r'#\1', tweet)
    
    # Normaliser les espaces
    tweet = re.sub(r'\s+', ' ', tweet)
    
    return tweet.strip()

def load_sentiment140_data():
    """Charge les données Sentiment140 (télécharge si nécessaire)"""
    sample_path = download_sentiment140_dataset()
    
    if sample_path and os.path.exists(sample_path):
        print("📊 Chargement de l'échantillon Sentiment140...")
        df = pd.read_csv(sample_path)
        
        # Ajouter des colonnes calculées
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['processed_text'] = df['text'].apply(preprocess_tweet)
        
        return df
    else:
        print("❌ Impossible de charger le dataset")
        return create_fallback_data()

def create_fallback_data():
    """Crée des données de fallback si le téléchargement échoue"""
    print("⚠️ Utilisation de données simulées")
    
    sample_texts = [
        "I love this movie! It's absolutely amazing and beautiful.",
        "This is the worst experience ever. Completely disappointed.",
        "Great product, highly recommend it to everyone!",
        "Not satisfied with the quality, could be much better.",
        "Amazing service and wonderful customer support!",
        "Terrible quality, waste of money and time.",
        "Pretty good overall, meets my expectations.",
        "Awful experience, will never buy again.",
        "Fantastic! Exceeded all my expectations completely.",
        "Poor service, very disappointing and frustrating."
    ] * 1000  # Répéter pour avoir plus de données
    
    sentiments = [1, 0] * 5000  # Alterné pour équilibrer
    
    df = pd.DataFrame({
        'target': sentiments[:len(sample_texts)],
        'text': sample_texts,
        'text_length': [len(text) for text in sample_texts],
        'word_count': [len(text.split()) for text in sample_texts]
    })
    
    df['processed_text'] = df['text'].apply(preprocess_tweet)
    
    return df

def analyze_text_statistics(df):
    """Analyse les statistiques textuelles du vrai dataset"""
    stats = {
        'total_texts': len(df),
        'positive_count': (df['target'] == 1).sum(),
        'negative_count': (df['target'] == 0).sum(),
        'avg_length': df['text_length'].mean(),
        'median_length': df['text_length'].median(),
        'min_length': df['text_length'].min(),
        'max_length': df['text_length'].max(),
        'std_length': df['text_length'].std(),
        'avg_words': df['word_count'].mean(),
        'median_words': df['word_count'].median(),
        'total_words': df['word_count'].sum()
    }
    
    return stats

def get_word_frequencies(df, top_n=20):
    """Calcule les fréquences des mots du vrai dataset"""
    all_words = []
    
    # Utiliser les textes prétraités
    for text in df['processed_text'].dropna():
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    # Filtrer les mots trop courts et les mots vides courants
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'a', 'an'}
    
    filtered_words = [word for word in all_words if len(word) > 2 and word not in stop_words]
    
    word_freq = Counter(filtered_words)
    return word_freq.most_common(top_n)

def create_wordcloud_from_data(df, sentiment=None):
    """Génère un WordCloud à partir du vrai dataset"""
    
    if sentiment is not None:
        texts = df[df['target'] == sentiment]['processed_text'].dropna()
        title = f"WordCloud - Sentiment {'Positif' if sentiment == 1 else 'Négatif'}"
    else:
        texts = df['processed_text'].dropna()
        title = "WordCloud - Tous les sentiments"
    
    # Combiner tous les textes
    all_text = ' '.join(texts.astype(str))
    
    # Créer le WordCloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(all_text)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)
    
    return fig

def get_temporal_analysis(df):
    """Analyse temporelle si les données de date sont disponibles"""
    if 'date' in df.columns:
        try:
            # Parser les dates (format Twitter)
            df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
            df['hour'] = df['parsed_date'].dt.hour
            df['day_of_week'] = df['parsed_date'].dt.day_name()
            
            return {
                'by_hour': df.groupby('hour')['target'].mean(),
                'by_day': df.groupby('day_of_week')['target'].mean(),
                'date_range': (df['parsed_date'].min(), df['parsed_date'].max())
            }
        except:
            return None
    return None

def get_basic_stats():
    """Retourne les statistiques de base du dataset"""
    return {
        'total_samples': 1600000,
        'sample_size': 50000,
        'positive_samples': 25000,
        'negative_samples': 25000,
        'source': 'Sentiment140 (Stanford)',
        'language': 'English',
        'platform': 'Twitter'
    }

def load_sample_data():
    """Fonction de compatibilité - redirige vers load_sentiment140_data"""
    return load_sentiment140_data()

def create_sentiment_distribution():
    """Crée un graphique de distribution des sentiments"""
    df = load_sentiment140_data()
    if df is not None:
        sentiment_counts = df['target'].value_counts().sort_index()
        return {
            'labels': ['Négatif', 'Positif'],
            'values': sentiment_counts.values,
            'percentages': (sentiment_counts / len(df) * 100).round(1)
        }
    else:
        # Données par défaut si le chargement échoue
        return {
            'labels': ['Négatif', 'Positif'],
            'values': [25000, 25000],
            'percentages': [50.0, 50.0]
        }