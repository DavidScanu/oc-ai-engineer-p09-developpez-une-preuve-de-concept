import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import sys
import os
import json
from utils.model_manager import ModelManager, format_improvement, get_performance_status, DISTILBERT_BASELINE
from sklearn.metrics import confusion_matrix


# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_manager import ModelManager, format_improvement, get_performance_status

st.set_page_config(
    page_title="Métriques des Modèles",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_model_manager():
    """Cache le gestionnaire de modèles"""
    return ModelManager()

def create_comparison_chart(comparison_df):
    """Crée un graphique de comparaison multi-modèles"""
    # Préparer les données pour le graphique
    metrics = ['Accuracy', 'F1-Score', 'Précision', 'Rappel', 'ROC AUC']
    
    fig = go.Figure()
    
    colors = ['#ff7f0e', '#2ca02c', "#ee6b6b", '#9467bd']
    
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        model_name = row['Modèle']
        color = colors[i % len(colors)]
        
        # Couleur spéciale pour le baseline
        if row['Type'] == 'baseline':
            color = '#1f77b4'
            model_name += ' (Baseline)'
        
        fig.add_trace(go.Scatter(
            x=metrics,
            y=[row[metric] for metric in metrics],
            mode='lines+markers',
            name=model_name,
            line=dict(color=color, width=3),
            marker=dict(size=8),
            text=[f'{row[metric]:.3f}' for metric in metrics],
            textposition='top center'
        ))
    
    fig.update_layout(
        title='Comparaison des Performances des Modèles',
        title_font_size=18,
        xaxis_title='Métriques',
        yaxis_title='Score',
        yaxis=dict(range=[0.7, 1.0]),
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    
    return fig

def create_improvement_chart(comparison):
    """Crée un graphique des améliorations"""
    improvements = comparison['improvements']
    
    metrics = list(improvements.keys())
    values = [improvements[metric]['percentage'] for metric in metrics]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f'{v:+.1f}%' for v in values],
            textposition='outside'
        )
    ])
    
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    fig.update_layout(
        title='Améliorations vs Baseline DistilBERT',
        title_font_size=16,
        xaxis_title='Métriques',
        yaxis_title='Amélioration (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def create_model_comparison_chart(selected_model, baseline):
    """Crée un graphique de comparaison avec le baseline (style barres groupées)"""
    
    # Préparer les données
    models_data = [
        {
            'Modèle': 'DistilBERT (Baseline)',
            'Accuracy': baseline['accuracy'],
            'F1-score': baseline['f1'],
            'Precision': baseline['precision'],
            'Recall': baseline['recall'],
            'ROC AUC': baseline['roc_auc']
        },
        {
            'Modèle': f'ModernBERT ({selected_model["model_id"]})',
            'Accuracy': selected_model['accuracy'],
            'F1-score': selected_model['f1'],
            'Precision': selected_model['precision'],
            'Recall': selected_model['recall'],
            'ROC AUC': selected_model['roc_auc']
        }
    ]
    
    df = pd.DataFrame(models_data)
    
    # Graphique en barres groupées
    fig = go.Figure()
    
    colors = ['#ff7f0e', '#2ca02c']  # Orange et vert
    metrics = ['Accuracy', 'F1-score', 'Precision', 'Recall', 'ROC AUC']
    
    for i, model in enumerate(df['Modèle']):
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=[df.iloc[i][metric] for metric in metrics],
            marker_color=colors[i],
            text=[f'{df.iloc[i][metric]:.3f}' for metric in metrics],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Comparaison des Performances des Modèles',
        title_font_size=18,
        xaxis_title='Métriques',
        yaxis_title='Score',
        yaxis=dict(range=[0.75, 1.0]),
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig, df

def create_training_history_chart(df: pd.DataFrame):
    """
    Affiche les courbes par epoch :
    - Training Loss
    - Validation Accuracy
    - Validation ROC AUC
    """
    if df is None or df.empty:
        return go.Figure()

    # --- Agréger par epoch (moyenne par epoch) ---
    # Crée une nouvelle colonne 'epoch_int' pour éviter le conflit
    df["epoch_int"] = df["epoch"].round(0).astype(int)
    df_epoch = df.groupby("epoch_int").mean().reset_index()

    fig = go.Figure()

    # Training Loss
    if "loss" in df_epoch.columns:
        fig.add_trace(go.Scatter(
            x=df_epoch["epoch_int"],
            y=df_epoch["loss"],
            mode="lines",
            name="Training Loss",
            line=dict(color="royalblue", width=2)
        ))

    # Validation Accuracy
    if "eval_accuracy" in df_epoch.columns and df_epoch["eval_accuracy"].notna().any():
        fig.add_trace(go.Scatter(
            x=df_epoch["epoch_int"],
            y=df_epoch["eval_accuracy"],
            mode="lines",
            name="Validation Accuracy",
            line=dict(color="darkorange", width=2),
            yaxis="y2"
        ))

    # Validation ROC AUC
    if "eval_roc_auc" in df_epoch.columns and df_epoch["eval_roc_auc"].notna().any():
        fig.add_trace(go.Scatter(
            x=df_epoch["epoch_int"],
            y=df_epoch["eval_roc_auc"],
            mode="lines",
            name="Validation ROC AUC",
            line=dict(color="green", width=2),
            yaxis="y2"
        ))

    # Layout dual-axis
    fig.update_layout(
        title="Historique d'entraînement (par Epoch)",
        xaxis_title="Epoch",
        yaxis=dict(title="Training Loss"),
        yaxis2=dict(title="Validation Metrics", overlaying="y", side="right"),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )

    return fig

def create_confusion_matrix_chart(df: pd.DataFrame):
    """
    Crée une matrice de confusion à partir du DataFrame df
    """
    if df.empty:
        return go.Figure()

    y_true = df["true_labels"]
    y_pred = df["predictions"]

    labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        colorscale="Blues",
        showscale=True,
        reversescale=False
    )

    fig.update_layout(
        title="Matrice de Confusion",
        xaxis_title="Prédiction",
        yaxis_title="Valeur Réelle"
    )

    return fig


def main():
    st.title("📈 Métriques et Comparaison des Modèles")
    st.markdown("---")
    
    # Initialiser le gestionnaire de modèles
    manager = load_model_manager()
    
    # Découvrir les modèles disponibles
    available_models = manager.discover_models()
    
    st.subheader("🔍 Modèles Disponibles")
    
    if not available_models:
        st.warning("⚠️ Aucun modèle ModernBERT trouvé dans le dossier models/")
        st.info("💡 Ajoutez un modèle dans le format : `models/modernbert-sentiment-yyyymmdd_hhmm/`")
        st.stop()
    
    # Sélecteur de modèle
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_options = {f"{model['model_id']} ({model['training_date']})": model for model in available_models}
        selected_model_key = st.selectbox(
            "Choisissez un modèle à analyser :",
            options=list(model_options.keys()),
            index=0
        )
        selected_model = model_options[selected_model_key]
    
    with col2:
        st.metric("Modèles disponibles", len(available_models))
        
        # Meilleur modèle
        best_model = manager.get_best_model('roc_auc')
        if best_model:
            is_best = best_model['model_id'] == selected_model['model_id']
            st.metric(
                "Statut", 
                "✔️ Meilleur" if is_best else "📊 Standard",
                help="Basé sur le ROC AUC"
            )
    
    # Comparaison avec baseline
    comparison = manager.compare_with_baseline(selected_model)
    status_color, status_message = get_performance_status(comparison)
    
    # Affichage du statut
    if status_color == "success":
        st.success(status_message)
    elif status_color == "info":
        st.info(status_message)
    elif status_color == "warning":
        st.warning(status_message)
    else:
        st.error(status_message)
    
    st.markdown("---")
    
    # Métriques détaillées
    st.subheader(f"📊 Performances Détaillées - {selected_model['model_id']}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_info = [
        ("Accuracy", "accuracy", "Précision globale"),
        ("F1-Score", "f1", "Moyenne harmonique"),
        ("Précision", "precision", "Vrais positifs"),
        ("Rappel", "recall", "Sensibilité"),
        ("ROC AUC", "roc_auc", "Discrimination")
    ]
    
    for i, (label, key, help_text) in enumerate(metrics_info):
        with [col1, col2, col3, col4, col5][i]:
            model_value = selected_model.get(key, 0)
            improvement = comparison['improvements'].get(key, {})
            delta_value = improvement.get('absolute', 0)
            
            st.metric(
                label,
                f"{model_value:.3f}",
                delta=f"{delta_value:+.3f}",
                delta_color="normal" if delta_value >= 0 else "inverse",
                help=help_text
            )


    # Informations techniques
    st.markdown("---")
    st.subheader("🔧 Informations Techniques")
    
    # Première ligne - Modèle
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Architecture", selected_model.get('architecture', 'N/A'))
    with col2:
        training_time = selected_model.get('training_time_minutes', 0)
        st.metric("Temps d'entraînement", f"{training_time:.1f} min")
    with col3:
        total_params = selected_model.get('total_params', 0)
        st.metric("Paramètres totaux", f"{total_params:,}")
    with col4:
        trainable_params = selected_model.get('trainable_params', 0)
        st.metric("Paramètres entraînables", f"{trainable_params:,}")
    
    # Deuxième ligne - Dataset et entraînement
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        epochs = selected_model.get('epochs_completed', 'N/A')
        st.metric("Époques", f"{epochs}")
    with col2:
        # Taille totale du dataset
        dataset_size = selected_model.get('total_samples', 0)
        if dataset_size > 0:
            st.metric("Dataset total", f"{dataset_size:,}")
        else:
            st.metric("Dataset total", "N/A")
    with col3:
        # Échantillons d'entraînement
        train_samples = selected_model.get('train_samples', 0)
        if train_samples > 0:
            st.metric("Échantillons train", f"{train_samples:,}")
        else:
            st.metric("Échantillons train", "N/A")
    with col4:
        # Équilibre des classes
        positive_samples = selected_model.get('positive_samples', 0)
        negative_samples = selected_model.get('negative_samples', 0)
        
        if positive_samples > 0 and negative_samples > 0:
            balance_ratio = positive_samples / (positive_samples + negative_samples) * 100
            st.metric("Équilibre pos/neg", f"{balance_ratio:.0f}%/{100-balance_ratio:.0f}%")
        else:
            st.metric("Équilibre pos/neg", "N/A")


    # Historique d'entraînement
    st.markdown("---")
    st.subheader("📉 Historique d'Entraînement")


    # Deuxième ligne - Dataset et entraînement
    col1, col2 = st.columns(2)
    
    with col1:

        if selected_model:
            df_history = manager.load_training_history(selected_model["model_path"])
            fig = create_training_history_chart(df_history)
            st.plotly_chart(fig, width='stretch')

    with col2:

        # Dans votre fonction main(), après avoir choisi le modèle :
        df_test_preds = manager.load_test_predictions(selected_model)
        fig_cm = create_confusion_matrix_chart(df_test_preds)
        st.plotly_chart(fig_cm, width='stretch')


    # Graphiques de comparaison
    st.markdown("---")
    st.subheader("🔄 Comparaison avec le Baseline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_comparison, df_comparison = create_model_comparison_chart(selected_model, manager.baseline)
        st.plotly_chart(fig_comparison, width='stretch')
    
    with col2:
        st.markdown("**Améliorations vs DistilBERT :**")
        
        improvements = comparison['improvements']
        
        for metric, improvement_data in improvements.items():
            improvement_pct = improvement_data['percentage']
            st.write(f"• **{metric.title()}**: {improvement_pct:+.1f}%")
        
        # Message conditionnel selon les performances
        if comparison['summary']['is_better']:
            st.success("✅ Toutes les métriques sont améliorées !")
        elif comparison['summary']['metrics_improved'] > 0:
            st.warning(f"⚠️ {comparison['summary']['metrics_improved']}/{comparison['summary']['total_metrics']} métriques améliorées")
        else:
            st.error("❌ Aucune métrique améliorée vs baseline")
    
    # Graphique des améliorations individuelles
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Détail des Améliorations")
        fig_improvements = create_improvement_chart(comparison)
        st.plotly_chart(fig_improvements, width='stretch')
        
        # Résumé des améliorations
        summary = comparison['summary']
        st.write(f"**Amélioration moyenne :** {summary['avg_improvement']:+.1f}%")
        st.write(f"**Métriques améliorées :** {summary['metrics_improved']}/{summary['total_metrics']}")
    
    with col2:
        st.subheader("📊 Comparaison Multi-Modèles")
        
        # Inclure tous les modèles dans la comparaison
        comparison_df = manager.get_comparison_dataframe()
        fig_comparison_all = create_comparison_chart(comparison_df)
        st.plotly_chart(fig_comparison_all, width='stretch')


    # Tableau de comparaison détaillé
    st.markdown("---")
    st.subheader("📋 Tableau de Comparaison Complet")

    # Formater le DataFrame pour l'affichage
    display_df = comparison_df.copy()

    # Convertir toutes les colonnes numériques, remplacer les non-numériques par NaN
    numeric_columns = ['Accuracy', 'F1-Score', 'Précision', 'Rappel', 'ROC AUC', 'Temps (min)']
    for col in numeric_columns:
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(3)

    # Styliser le tableau
    def highlight_best(s):
        """Surligne les meilleures valeurs"""
        if s.name in numeric_columns:
            max_val = s.max()
            return ['background-color: #e8f5e8' if v == max_val else '' for v in s]
        return [''] * len(s)

    def highlight_baseline(row):
        """Surligne la ligne baseline"""
        if row['Type'] == 'baseline':
            return ['background-color: #fff3cd'] * len(row)
        return [''] * len(row)

    # styled_df = display_df.style.apply(highlight_best, axis=0).apply(highlight_baseline, axis=1)
    # st.dataframe(styled_df, width='stretch')

    styled_df = display_df.style.format({col: "{:.3f}" for col in numeric_columns}) \
                                .apply(highlight_best, axis=0) \
                                .apply(highlight_baseline, axis=1)
    st.dataframe(styled_df, width='stretch')


    
    # Analyse détaillée
    with st.expander("🔍 Analyse Détaillée"):
        st.markdown(f"""
        **Modèle analysé :** {selected_model['model_id']}
        
        **Points forts :**
        """)
        
        for metric, improvement in comparison['improvements'].items():
            if improvement['better']:
                st.write(f"• **{metric.title()}** : {format_improvement(improvement)} vs baseline")
        
        st.markdown("**Points d'amélioration :**")
        
        for metric, improvement in comparison['improvements'].items():
            if not improvement['better']:
                st.write(f"• **{metric.title()}** : {format_improvement(improvement)} vs baseline")
        
        # Recommandations
        st.markdown("**Recommandations :**")
        
        if comparison['summary']['is_better']:
            if comparison['summary']['significant_improvement']:
                st.write("🎯 **Modèle recommandé pour la production**")
                st.write("• Performance supérieure au baseline sur la majorité des métriques")
                st.write("• Amélioration significative détectée")
            else:
                st.write("📈 **Modèle prometteur mais perfectible**")
                st.write("• Légère amélioration vs baseline")
                st.write("• Considérer des optimisations supplémentaires")
        else:
            st.write("⚠️ **Optimisation requise**")
            st.write("• Performance inférieure au baseline")
            st.write("• Réviser l'architecture ou les hyperparamètres")
            st.write("• Analyser les causes de sous-performance")
    



    # Résumé exécutif dynamique
    st.markdown("---")
    st.subheader("📋 Résumé Exécutif")
    
    # Calculer les métriques clés pour le résumé
    summary = comparison['summary']
    selected_metrics = {
        'accuracy': selected_model.get('accuracy', 0),
        'f1': selected_model.get('f1', 0),
        'precision': selected_model.get('precision', 0),
        'recall': selected_model.get('recall', 0),
        'roc_auc': selected_model.get('roc_auc', 0)
    }
    
    baseline_metrics = {
        'accuracy': manager.baseline['accuracy'],
        'f1': manager.baseline['f1'],
        'precision': manager.baseline['precision'],
        'recall': manager.baseline['recall'],
        'roc_auc': manager.baseline['roc_auc']
    }
    
    # Informations techniques
    training_time = selected_model.get('training_time_minutes', 0)
    total_params = selected_model.get('total_params', 0)
    trainable_params = selected_model.get('trainable_params', 0)
    trainable_ratio = (trainable_params / total_params * 100) if total_params > 0 else 0
    
    # Messages conditionnels selon les performances
    if summary['is_better'] and summary['significant_improvement']:
        # Modèle excellent
        st.success(f"""
        **🎯 MODÈLE RECOMMANDÉ POUR LA PRODUCTION**
        
        **Modèle :** {selected_model['model_id']} ({selected_model['training_date']})
        
        **Performance :**
        ✅ **ROC AUC excellent** : {selected_metrics['roc_auc']:.1%} (+{comparison['improvements']['roc_auc']['percentage']:+.1f}% vs baseline)
        ✅ **Accuracy supérieure** : {selected_metrics['accuracy']:.1%} (+{comparison['improvements']['accuracy']['percentage']:+.1f}% vs baseline)
        ✅ **F1-Score optimal** : {selected_metrics['f1']:.1%} (+{comparison['improvements']['f1']['percentage']:+.1f}% vs baseline)
        ✅ **Amélioration moyenne** : {summary['avg_improvement']:+.1f}% sur toutes les métriques
        
        **Efficacité :**
        ⚡ **Entraînement efficace** : {training_time:.1f} minutes seulement
        🎯 **Paramètres optimisés** : {trainable_params:,} paramètres fine-tunés ({trainable_ratio:.3f}% du total)
        
        **Recommandation :** Déploiement immédiat recommandé
        """)
        
    elif summary['is_better']:
        # Modèle avec amélioration modérée
        st.info(f"""
        **📈 MODÈLE PROMETTEUR - OPTIMISATION POSSIBLE**
        
        **Modèle :** {selected_model['model_id']} ({selected_model['training_date']})
        
        **Performance :**
        ✅ **ROC AUC** : {selected_metrics['roc_auc']:.1%} (+{comparison['improvements']['roc_auc']['percentage']:+.1f}% vs baseline)
        ✅ **Accuracy** : {selected_metrics['accuracy']:.1%} (+{comparison['improvements']['accuracy']['percentage']:+.1f}% vs baseline)
        ✅ **Métriques améliorées** : {summary['metrics_improved']}/{summary['total_metrics']} au-dessus du baseline
        📊 **Amélioration moyenne** : {summary['avg_improvement']:+.1f}%
        
        **Points forts :**
        """)
        
        # Lister les métriques améliorées
        for metric, improvement in comparison['improvements'].items():
            if improvement['better']:
                st.write(f"   • **{metric.title()}** : {improvement['percentage']:+.1f}% d'amélioration")
        
        st.info(f"""
        **Recommandation :** Considérer pour production après tests A/B
        """)
        
    elif summary['avg_improvement'] > -2.0:
        # Performance similaire
        st.warning(f"""
        **⚠️ PERFORMANCE SIMILAIRE AU BASELINE**
        
        **Modèle :** {selected_model['model_id']} ({selected_model['training_date']})
        
        **Performance :**
        📊 **ROC AUC** : {selected_metrics['roc_auc']:.1%} ({comparison['improvements']['roc_auc']['percentage']:+.1f}% vs baseline)
        📊 **Accuracy** : {selected_metrics['accuracy']:.1%} ({comparison['improvements']['accuracy']['percentage']:+.1f}% vs baseline)
        ⚖️ **Différence moyenne** : {summary['avg_improvement']:+.1f}% (non significative)
        
        **Analyse :**
        • Pas d'amélioration significative détectée
        • Performance équivalente au baseline DistilBERT
        • Coût computationnel plus élevé sans bénéfice clair
        
        **Recommandation :** Continuer l'optimisation ou rester sur DistilBERT
        """)
        
    else:
        # Performance inférieure
        st.warning(f"""
        **🚨 OPTIMISATION REQUISE - NON RECOMMANDÉ POUR PRODUCTION**
        
        **Modèle :** {selected_model['model_id']} ({selected_model['training_date']})
        
        **Performance :**
        - ❌ **ROC AUC inférieur** : {selected_metrics['roc_auc']:.1%} ({comparison['improvements']['roc_auc']['percentage']:+.1f}% vs baseline)
        - ❌ **Accuracy réduite** : {selected_metrics['accuracy']:.1%} ({comparison['improvements']['accuracy']['percentage']:+.1f}% vs baseline)
        - 📉 **Dégradation moyenne** : {summary['avg_improvement']:+.1f}% sur les métriques
        
        **Problèmes identifiés :**
        - Sous-performance sur les métriques clés
        - Pas d'amélioration significative par rapport au baseline DistilBERT
        - Ratio d'entraînement faible : {trainable_ratio:.3f}% des paramètres entraînables

        **Causes possibles :**
        - Hyperparamètres inadéquats (learning rate: trop élevé/bas)
        - Gel excessif ({trainable_ratio:.3f}% paramètres entraînables)
        
        **Actions recommandées :**
        1. 🔄 Ré-entraînement avec hyperparamètres optimisés
        2. 🎯 Dégel de plus de couches BERT
        3. 📈 Augmentation du temps d'entraînement
        4. 🔍 Analyse approfondie des erreurs de prédiction
        
        **Décision :** Rester sur DistilBERT jusqu'à amélioration
        """)

        st.success("🚀 **Prochaines étapes :** Optimisation et ré-entraînement du modèle")
    
    # Section technique détaillée
    with st.expander("🔧 Détails Techniques du Modèle"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Configuration :**")
            st.write(f"• Architecture : {selected_model.get('architecture', 'N/A')}")
            st.write(f"• Vocabulaire : {selected_model.get('vocab_size', 0):,} tokens")
            st.write(f"• Paramètres totaux : {total_params:,}")
            st.write(f"• Paramètres entraînés : {trainable_params:,}")
            st.write(f"• Ratio entraînable : {trainable_ratio:.3f}%")
        
        with col2:
            st.markdown("**Entraînement :**")
            st.write(f"• Temps total : {training_time:.1f} minutes")
            st.write(f"• Époques : {selected_model.get('epochs_completed', 'N/A')}")
            st.write(f"• Steps totaux : {selected_model.get('total_steps', 'N/A'):,}")
            st.write(f"• Loss finale : {selected_model.get('loss', 0):.4f}")
            st.write(f"• Meilleure métrique : {selected_model.get('best_metric', 'N/A')}")
        
        with col3:
            st.markdown("**Comparaison Baseline :**")
            st.write(f"• DistilBERT Accuracy : {baseline_metrics['accuracy']:.1%}")
            st.write(f"• DistilBERT F1 : {baseline_metrics['f1']:.1%}")
            st.write(f"• DistilBERT ROC AUC : {baseline_metrics['roc_auc']:.1%}")
            st.write(f"• Écart moyen : {summary['avg_improvement']:+.1f}%")
            st.write(f"• Métriques ✅ : {summary['metrics_improved']}/{summary['total_metrics']}")
    
    # Recommandations d'amélioration si nécessaire
    if not summary['is_better'] or not summary['significant_improvement']:
        with st.expander("💡 Stratégies d'Amélioration"):
            st.markdown("""
            **Approches recommandées pour optimiser les performances :**
            
            ### 🎯 Optimisation des Hyperparamètres
            ```python
            # Configuration A - Dégel progressif
            learning_rates = {
                'classifier': 2e-5,
                'bert_layers': [1e-6, 5e-6, 1e-5, 2e-5]  # Graduel
            }
            
            # Configuration B - Learning rate adaptatif
            scheduler = 'cosine_with_restarts'
            warmup_ratio = 0.1
            ```
            
            ### 🔄 Stratégies d'Entraînement
            - **Dégel progressif** : Commencer par le classifier, puis dégeler couche par couche
            - **Curriculum learning** : Entraîner d'abord sur exemples faciles
            - **Mixup/CutMix** : Augmentation de données pour robustesse
            
            ### 📊 Analyse et Monitoring
            - **Validation curves** : Surveiller overfitting en temps réel
            - **Gradient analysis** : Vérifier la propagation des gradients
            - **Layer-wise metrics** : Analyser l'apprentissage par couche
            
            ### 🔍 Diagnostic Avancé
            - **Error analysis** : Identifier les types d'erreurs récurrentes
            - **Attention visualization** : Comprendre le focus du modèle
            - **Embedding analysis** : Vérifier la qualité des représentations
            """)
    
    # Call-to-action selon le contexte
    st.markdown("---")
    
    if summary['is_better'] and summary['significant_improvement']:
        st.success("🚀 **Prêt pour le déploiement !** Ce modèle peut être mis en production.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("📋 **Prochaines étapes :**\n1. Tests d'intégration\n2. Validation métier\n3. Déploiement progressif")
        with col2:
            st.info("📊 **Monitoring recommandé :**\n1. Drift detection\n2. Performance tracking\n3. A/B testing continu")
    
    else:
        st.warning("🔄 **Optimisation nécessaire** avant mise en production.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("🎯 **Actions prioritaires :**\n1. Ré-entraînement optimisé\n2. Validation croisée\n3. Analyse d'erreurs")
        with col2:
            st.info("⏱️ **Timeline suggérée :**\n1. Optimisation : 1-2 semaines\n2. Tests : 1 semaine\n3. Validation : 1 semaine")








    # Export des données
    st.markdown("---")
    
    with st.expander("💾 Export des Données"):
        st.markdown("**Données de comparaison au format JSON :**")
        
        export_data = {
            'selected_model': selected_model,
            'baseline': manager.baseline,
            'comparison': comparison,
            'all_models': available_models
        }
        
        st.json(export_data)
        
        # Bouton de téléchargement CSV
        csv_data = comparison_df.to_csv(index=False)
        st.download_button(
            label="📊 Télécharger le tableau en CSV",
            data=csv_data,
            file_name=f"model_comparison_{selected_model['model_id']}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()