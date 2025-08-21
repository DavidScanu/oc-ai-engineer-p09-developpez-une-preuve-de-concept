#!/usr/bin/env python3
"""
Script de diagnostic des imports
"""

import sys
import os

# Ajouter le chemin de l'application au PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test tous les imports nécessaires"""
    
    print("🔍 Test des imports...")
    
    try:
        print("1. Test utils.data_analysis...")
        from utils.data_analysis import (
            download_sentiment140_dataset, 
            load_sentiment140_data,
            get_basic_stats,
            analyze_text_statistics,
            get_word_frequencies
        )
        print("   ✅ utils.data_analysis OK")
    except ImportError as e:
        print(f"   ❌ Erreur utils.data_analysis: {e}")
    
    try:
        print("2. Test utils.visualizations...")
        from utils.visualizations import (
            create_accessible_colors,
            create_confusion_matrix_plot,
            create_roc_curve
        )
        print("   ✅ utils.visualizations OK")
    except ImportError as e:
        print(f"   ❌ Erreur utils.visualizations: {e}")
    
    try:
        print("3. Test utils.model_utils...")
        from utils.model_utils import SentimentPredictor, load_model_info
        print("   ✅ utils.model_utils OK")
    except ImportError as e:
        print(f"   ❌ Erreur utils.model_utils: {e}")
    
    print("\n📁 Structure des fichiers:")
    for root, dirs, files in os.walk("utils"):
        level = root.replace("utils", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

if __name__ == "__main__":
    test_imports()