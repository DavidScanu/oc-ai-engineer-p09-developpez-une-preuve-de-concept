import streamlit as st
import requests
import os
from tqdm import tqdm

def download_with_progress(url, local_path, description="Téléchargement"):
    """Télécharge un fichier avec barre de progression Streamlit"""
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    downloaded = 0
    
    with open(local_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    progress = downloaded / total_size
                    progress_bar.progress(progress)
                    status_text.text(f"{description}: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB")
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ {description} terminé!")
    
    return True