import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_accessible_colors():
    """Retourne une palette de couleurs accessible pour daltoniens"""
    return {
        'positive': '#2ca02c',  # Vert
        'negative': '#d62728',  # Rouge
        'neutral': '#ff7f0e',   # Orange
        'primary': '#1f77b4',   # Bleu
        'secondary': '#9467bd'  # Violet
    }

def create_sentiment_distribution():
    """Crée un graphique de distribution des sentiments"""
    # Cette fonction peut être importée depuis data_analysis si nécessaire
    pass

def create_confusion_matrix_plot(y_true, y_pred, labels=['Négatif', 'Positif']):
    """Crée une matrice de confusion interactive"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Matrice de Confusion",
        title_font_size=18,
        xaxis_title="Prédictions",
        yaxis_title="Vraies Valeurs",
        font=dict(size=12)
    )
    
    return fig

def create_roc_curve(y_true, y_scores):
    """Crée une courbe ROC interactive"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})',
        line=dict(color='#2ca02c', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Ligne de base',
        line=dict(color='#d62728', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Courbe ROC',
        title_font_size=18,
        xaxis_title='Taux de Faux Positifs',
        yaxis_title='Taux de Vrais Positifs',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def create_metrics_comparison(metrics_dict):
    """Crée un graphique de comparaison des métriques"""
    df = pd.DataFrame([metrics_dict])
    
    fig = go.Figure()
    
    colors = create_accessible_colors()
    
    for i, (metric, value) in enumerate(metrics_dict.items()):
        fig.add_trace(go.Bar(
            x=[metric],
            y=[value],
            name=metric,
            marker_color=list(colors.values())[i % len(colors)],
            text=f'{value:.3f}',
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Métriques de Performance',
        title_font_size=18,
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def create_prediction_distribution(predictions):
    """Crée un graphique de distribution des prédictions"""
    sentiment_counts = pd.Series(predictions).value_counts()
    
    colors = create_accessible_colors()
    color_map = {'Positif': colors['positive'], 'Négatif': colors['negative']}
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Distribution des Prédictions",
        color_discrete_map=color_map
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=12
    )
    
    fig.update_layout(
        title_font_size=18,
        font=dict(size=12)
    )
    
    return fig