import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os

class SentimentPredictor:
    """Classe pour gérer les prédictions de sentiment"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
    def load_model(self):
        """Charge le modèle et le tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            self.loaded = True
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def preprocess_text(self, text):
        """Prétraite le texte selon les standards d'entraînement"""
        if not isinstance(text, str):
            return ""
        
        text = str(text).strip()
        
        # Remplacer les URLs
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # Remplacer les mentions
        text = re.sub(r'@\w+', '[USER]', text)
        
        # Normaliser les hashtags
        text = re.sub(r'#(\w+)', r'#\1', text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def predict(self, text):
        """Effectue une prédiction de sentiment"""
        if not self.loaded:
            raise ValueError("Modèle non chargé. Appelez load_model() d'abord.")
        
        # Prétraitement
        processed_text = self.preprocess_text(text)
        
        # Tokenisation
        inputs = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'sentiment': 'Positif' if predicted_class == 1 else 'Négatif',
            'confidence': confidence,
            'probabilities': {
                'Négatif': probabilities[0][0].item(),
                'Positif': probabilities[0][1].item()
            },
            'processed_text': processed_text
        }

def load_model_info(model_path):
    """Charge les informations sur le modèle"""
    info_path = os.path.join(model_path, "model_info.json")
    
    try:
        import json
        with open(info_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "model_name": "ModernBERT-base",
            "architecture": "ModernBERT",
            "task": "sentiment_classification",
            "total_params": "149M",
            "trainable_params": "1.5K"
        }