#!/usr/bin/env python3
"""
Script de vérification et téléchargement du dataset Sentiment140
"""

import os
import sys
from utils.data_analysis import download_sentiment140_dataset, load_sentiment140_data

def main():
    print("🔍 Vérification du dataset Sentiment140...")
    
    # Vérifier si le dataset existe
    data_dir = "data"
    sample_file = os.path.join(data_dir, "sentiment140_sample.csv")
    
    if os.path.exists(sample_file):
        print(f"✅ Dataset échantillon trouvé : {sample_file}")
        
        # Charger et vérifier
        df = load_sentiment140_data()
        if df is not None:
            print(f"📊 Dataset chargé avec succès : {len(df):,} échantillons")
            print(f"   - Positifs : {(df['target'] == 1).sum():,}")
            print(f"   - Négatifs : {(df['target'] == 0).sum():,}")
        else:
            print("❌ Erreur lors du chargement")
    else:
        print("⚠️ Dataset non trouvé, téléchargement en cours...")
        result = download_sentiment140_dataset(force_download=True)
        
        if result:
            print("✅ Dataset téléchargé et préparé avec succès!")
        else:
            print("❌ Échec du téléchargement")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())