import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Métriques de référence DistilBERT (baseline fixe)
DISTILBERT_BASELINE = {
    'model_name': 'DistilBERT',
    'model_type': 'baseline',
    'accuracy': 0.829,
    'f1': 0.827,
    'precision': 0.838,
    'recall': 0.816,
    'roc_auc': 0.899,
    'training_date': '2024-01-01',
    'notes': 'Baseline de référence'
}

class ModelManager:
    """Gestionnaire pour les modèles et leurs métriques"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.baseline = DISTILBERT_BASELINE
    
    def discover_models(self) -> List[Dict]:
        """Découvre automatiquement tous les modèles disponibles"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for item in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, item)
            
            if os.path.isdir(model_path) and item.startswith('modernbert-sentiment-'):
                model_info = self._load_model_info(model_path)
                if model_info:
                    model_info['model_id'] = item
                    model_info['model_path'] = model_path
                    models.append(model_info)
        
        # Trier par date (plus récent en premier)
        models.sort(key=lambda x: x.get('training_date', ''), reverse=True)
        
        return models


    def _load_dataset_config(self, model_path: str) -> Dict:
        """Charge la configuration du dataset"""
        try:
            dataset_config_path = os.path.join(model_path, "configs", "dataset_config.json")
            if os.path.exists(dataset_config_path):
                with open(dataset_config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement de dataset_config.json: {e}")
        
        return {}

    def _load_model_info(self, model_path: str) -> Optional[Dict]:
        """Charge les informations d'un modèle spécifique"""
        try:
            # Chemins des fichiers de métriques
            metrics_dir = os.path.join(model_path, "metrics")
            model_dir = os.path.join(model_path, "model")
            
            if not os.path.exists(metrics_dir):
                return None
            
            info = {}
            
            # 1. Métriques de test
            test_results_path = os.path.join(metrics_dir, "test_results.json")
            if os.path.exists(test_results_path):
                with open(test_results_path, 'r') as f:
                    test_results = json.load(f)
                    info.update({
                        'accuracy': test_results.get('test_accuracy', 0),
                        'f1': test_results.get('test_f1', 0),
                        'precision': test_results.get('test_precision', 0),
                        'recall': test_results.get('test_recall', 0),
                        'roc_auc': test_results.get('test_roc_auc', 0),
                        'loss': test_results.get('test_loss', 0)
                    })
            
            # 2. Métriques d'entraînement
            training_metrics_path = os.path.join(metrics_dir, "training_metrics.json")
            if os.path.exists(training_metrics_path):
                with open(training_metrics_path, 'r') as f:
                    training_metrics = json.load(f)
                    info.update({
                        'training_time_minutes': training_metrics.get('training_time_minutes', 0),
                        'epochs_completed': training_metrics.get('epochs_completed', 0),
                        'total_steps': training_metrics.get('total_steps', 0),
                        'best_metric': training_metrics.get('best_metric', 0)
                    })
            
            # 3. Informations du modèle
            model_info_path = os.path.join(model_dir, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    info.update({
                        'model_name': model_info.get('model_name', 'ModernBERT'),
                        'architecture': model_info.get('architecture', 'ModernBERT-base'),
                        'total_params': model_info.get('total_params', 0),
                        'trainable_params': model_info.get('trainable_params', 0),
                        'vocab_size': model_info.get('vocab_size', 0)
                    })
            
            # 4. Extraire la date depuis le nom du dossier
            folder_name = os.path.basename(model_path)
            if 'modernbert-sentiment-' in folder_name:
                date_part = folder_name.replace('modernbert-sentiment-', '')
                try:
                    # Format: yyyymmdd_hhmm
                    parsed_date = datetime.strptime(date_part, '%Y%m%d_%H%M')
                    info['training_date'] = parsed_date.strftime('%Y-%m-%d %H:%M')
                except:
                    info['training_date'] = date_part
            
            # 4. Configuration du dataset
            dataset_config = self._load_dataset_config(model_path)
            if dataset_config:
                info.update({
                    'total_samples': dataset_config.get('total_samples', 0),
                    'train_samples': dataset_config.get('train_samples', 0),
                    'val_samples': dataset_config.get('val_samples', 0),
                    'test_samples': dataset_config.get('test_samples', 0),
                    'positive_samples': dataset_config.get('positive_samples', 0),
                    'negative_samples': dataset_config.get('negative_samples', 0),
                    'max_sequence_length': dataset_config.get('preprocessing', {}).get('max_sequence_length', 512)
                })

            info['model_type'] = 'modernbert'
            
            return info
            
        except Exception as e:
            print(f"Erreur lors du chargement de {model_path}: {e}")
            return None
    
    def get_best_model(self, metric='roc_auc') -> Optional[Dict]:
        """Retourne le meilleur modèle selon une métrique"""
        models = self.discover_models()
        
        if not models:
            return None
        
        return max(models, key=lambda x: x.get(metric, 0))
    
    def get_latest_model(self) -> Optional[Dict]:
        """Retourne le modèle le plus récent"""
        models = self.discover_models()
        
        if not models:
            return None
        
        return models[0]  # Déjà trié par date
    
    def compare_with_baseline(self, model_info: Dict) -> Dict:
        """Compare un modèle avec le baseline DistilBERT"""
        comparison = {
            'model': model_info,
            'baseline': self.baseline,
            'improvements': {},
            'summary': {}
        }
        
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        
        for metric in metrics:
            model_value = model_info.get(metric, 0)
            baseline_value = self.baseline.get(metric, 0)
            
            if baseline_value > 0:
                improvement = ((model_value - baseline_value) / baseline_value) * 100
                comparison['improvements'][metric] = {
                    'absolute': model_value - baseline_value,
                    'percentage': improvement,
                    'better': improvement > 0
                }
        
        # Résumé global
        improvements = [comp['percentage'] for comp in comparison['improvements'].values()]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        
        comparison['summary'] = {
            'avg_improvement': avg_improvement,
            'metrics_improved': positive_improvements,
            'total_metrics': len(metrics),
            'is_better': positive_improvements > len(metrics) // 2,
            'significant_improvement': avg_improvement > 2.0  # Seuil de 2%
        }
        
        return comparison
    
    def get_comparison_dataframe(self, models: List[Dict] = None) -> pd.DataFrame:
        """Retourne un DataFrame de comparaison"""
        if models is None:
            models = self.discover_models()
        
        # Ajouter le baseline
        all_models = [self.baseline] + models
        
        df_data = []
        for model in all_models:
            row = {
                'Modèle': model.get('model_name', 'Unknown'),
                'Type': model.get('model_type', 'unknown'),
                'Date': model.get('training_date', 'N/A'),
                'Accuracy': model.get('accuracy', 0),
                'F1-Score': model.get('f1', 0),
                'Précision': model.get('precision', 0),
                'Rappel': model.get('recall', 0),
                'ROC AUC': model.get('roc_auc', 0),
                'Temps (min)': model.get('training_time_minutes', 0) if model.get('model_type') != 'baseline' else 'N/A'
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """Récupère un modèle par son ID"""
        models = self.discover_models()
        
        for model in models:
            if model.get('model_id') == model_id:
                return model
        
        return None

def format_improvement(improvement_data: Dict) -> str:
    """Formate l'affichage d'une amélioration"""
    percentage = improvement_data['percentage']
    symbol = "📈" if percentage > 0 else "📉" if percentage < 0 else "➡️"
    
    return f"{symbol} {percentage:+.1f}%"

def get_performance_status(comparison: Dict) -> Tuple[str, str]:
    """Retourne le statut de performance (couleur, message)"""
    summary = comparison['summary']
    
    if summary['is_better'] and summary['significant_improvement']:
        return "success", "🎯 Performance supérieure - Recommandé pour production"
    elif summary['is_better']:
        return "info", "📈 Amélioration modérée - À considérer"
    elif summary['avg_improvement'] > -2.0:
        return "warning", "⚠️ Performance similaire - Optimisation possible"
    else:
        return "warning", "🚨 Performance inférieure - Optimisation requise"