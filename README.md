# 📝 Projet 9 – Développez une preuve de concept : Amélioration d’un modèle d’analyse de sentiment de tweets

> 🎓 OpenClassrooms • Parcours [AI Engineer](https://openclassrooms.com/fr/paths/795-ai-engineer) | 👋 *Étudiant* : [David Scanu](https://www.linkedin.com/in/davidscanu14/)

## 📌 Introduction

Pour le **Projet 9 – Développez une preuve de concept**, nous avons choisi de poursuivre et d’améliorer un travail amorcé lors du **Projet 7 – Réalisez une analyse de sentiments grâce au Deep Learning**. Ce dernier avait abouti à la mise en place d’un pipeline complet d’analyse de sentiment de tweets, utilisant notamment un modèle **DistilBERT** fine-tuné sur le jeu de données **Sentiment140**.  

## ✨ Objectif

L’objectif du Projet 9 est de **dépasser les performances obtenues précédemment** en explorant de nouvelles approches plus performantes et adaptées au contexte, avec un accent particulier sur le **fine-tuning de ModernBERT**. Cette démarche s’appuiera sur des comparaisons avec des modèles spécialisés RoBERTa et des LLMs généralistes en mode zero-shot, afin de valider la pertinence et la supériorité de la solution proposée.

## 📊 Jeu de données : Sentiment140

**Sentiment140** est un jeu de données composé de **1,6 million de tweets** annotés automatiquement comme **positifs (1)** ou **négatifs (0)**.  
Ses caractéristiques principales :  
- **Format** : CSV avec colonnes `target`, `ids`, `date`, `flag`, `user`, `text`  
- **Langue** : Anglais  
- **Particularité** : Capture les spécificités du langage Twitter (hashtags, mentions, abréviations, emojis).

Ce dataset est largement utilisé comme référence pour l’entraînement et l’évaluation de modèles d’analyse de sentiment.

## 🎯 Tâche : Classification binaire

La tâche consiste à prédire si un tweet exprime un **sentiment positif** ou **négatif**. 

Dans le **Projet 7**, notre modèle **DistilBERT** fine-tuné sur Sentiment140 a atteint les performances suivantes :  

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 0.829  |
| F1-score    | 0.827  |
| Precision   | 0.838  |
| Recall      | 0.816  |
| ROC AUC     | 0.899  |

Ces résultats constituent notre **baseline** pour évaluer les approches du Projet 9.

## 🧠 Approches

### 🔹 Classification zero-shot

La **classification zero-shot** est une approche où un modèle pré-entraîné peut classer un texte dans des catégories **sans avoir été spécifiquement entraîné** pour cette tâche ou ces labels. Elle exploite la compréhension linguistique acquise lors du pré-entraînement et repose souvent sur des modèles de type *transformer* ou des **LLMs** via des consignes (*prompts*).

#### Modèles RoBERTa spécialisés

- [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)  
- [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english)  

Ces modèles sont optimisés pour l’analyse de sentiment et pré-entraînés sur de grandes quantités de données Twitter ou textuelles générales.

#### LLM Claude AI

L’utilisation d’un **LLM** comme **Claude AI** pour la classification zero-shot consiste à exploiter ses capacités de compréhension via des *prompts* décrivant la tâche d'analse de sentimet et les classes que nous cherchons à prédire (positive/negative).  

Nous avons testé deux modèles Anthropic :
- `claude-3-haiku-20240307` (version rapide et économique)

### 🔹 Transfer learning et ModernBERT

Le **transfer learning** consiste à réutiliser un modèle pré-entraîné sur un large corpus pour l’adapter à une tâche spécifique. Pour notre tâche, nous exploitons **ModernBERT**, un modèle pré-entraîné sur plus de **2 000 milliards de tokens**, optimisé pour la vitesse d’inférence, l’efficacité mémoire et la gestion de longues séquences. En appliquant un fine-tuning sur **Sentiment140**, nous adaptons ModernBERT aux particularités du langage Twitter, avec pour objectif de **surpasser les performances obtenues avec DistilBERT**.

#### ModernBERT 

[ModernBERT](https://huggingface.co/docs/transformers/model_doc/modernbert) est un encodeur *transformer* de dernière génération, publié en décembre 2014, offrant un équilibre optimal entre **performance**, **vitesse d’inférence** et **efficacité mémoire**. Il se distingue par des résultats *state-of-the-art* sur le benchmark **GLUE** pour un modèle de sa taille, surpassant notamment **DeBERTaV3-base**, preuve de ses capacités avancées de compréhension linguistique.

Pré-entraîné sur **2 000 milliards de tokens** (texte et code) avec une **longueur de séquence native de 8 192 tokens**, il bénéficie d’une couverture linguistique exceptionnelle et d’une grande robustesse face au vocabulaire varié et informel des tweets. Son architecture intègre des innovations modernes — **GeGLU**, **RoPE positional embeddings**, **attention locale/alternée**, gestion native des **séquences non paddées** — optimisées pour le calcul GPU et la rapidité d’inférence.

En pratique, **ModernBERT** traite des batchs plus grands que ses concurrents (jusqu’à ×2 pour la version base) et est environ **deux fois plus rapide** que DeBERTaV3 sur des contextes courts ou longs. Cette efficacité permet de concilier **précision élevée** et **coût de calcul réduit**, un atout clé pour un déploiement en production.

Dans le cadre de notre projet, nous procédons à un **fine-tuning** sur le jeu de données **Sentiment140** afin d’adapter le modèle à la classification binaire de sentiments sur tweets. L’objectif est de dépasser les performances obtenues lors du Projet 7 avec DistilBERT, en tirant parti du pré-entraînement massif et des optimisations structurelles de ModernBERT pour obtenir des métriques supérieures tout en maintenant une excellente efficacité opérationnelle.

### 📈 Tableau comparatif des modèles

| Modèle                                          | Type          | Stratégie          | Points forts                                                                 | Limites prévues                                             |
|------------------------------------------------|--------------|--------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------|
| **cardiffnlp/twitter-roberta-base-sentiment-latest** | Transformer   | Zero-shot          | Spécialisé Twitter, bon équilibre précision/rapidité                         | Moins performant hors domaine Twitter                        |
| **siebert/sentiment-roberta-large-english**     | Transformer   | Zero-shot          | Grande capacité, robuste sur texte général                                   | Temps d’inférence plus élevé                                 |
| **Claude 3 Haiku 20240307**                     | LLM           | Zero-shot (API)    | Faible coût, rapidité, pas de prétraitement complexe                         | Dépendance API, pas optimisé pour tweets courts               |
| **ModernBERT (fine-tuning)**                    | Transformer   | Transfer learning  | Pré-entraînement massif (2T tokens), optimisé mémoire & vitesse, séquence longue | Nécessite entraînement, tuning hyperparamètres                |

> **Note** : les performances réelles seront mesurées sur le même jeu de test Sentiment140 afin d’assurer une comparaison équitable.

## 🧪 Démarche expérimentale

1. **Préparation des données**
   - Nettoyage et tokenisation adaptée à chaque modèle
   - Split : 80% train, 10% validation, 10% test

2. **Évaluation zero-shot**
   - RoBERTa et Claude AI évalués sans fine-tuning
   - Mesures : Accuracy, F1, Precision, Recall, ROC AUC

3. **Fine-tuning de ModernBERT**
   - Hyperparamètres optimisés (learning rate, batch size, epochs)
   - Entraînement sur GPU
   - Comparaison des performances sur le même set de test

4. **Analyse et comparaison**
   - Tableau comparatif des performances
   - Analyse d’erreurs et cas limites

## 📚 Références

1. Go, A., Bhayani, R., & Huang, L. (2009). [Twitter Sentiment Classification using Distant Supervision](http://help.sentiment140.com/for-students)  
2. Wolf, T. et al. (2020). [Transformers: State-of-the-Art Natural Language Processing](https://arxiv.org/abs/1910.03771)  
3. Answer.AI. (2024). [ModernBERT: Efficient Transformer for Long-Sequence NLP](https://huggingface.co/answerdotai/ModernBERT-base)

---

## 📊 Dashboard Streamlit

### Fonctionnalités principales

Le dashboard interactif permet de valider la preuve de concept à travers quatre composants :

- **Analyse exploratoire** : Statistiques du dataset Sentiment140, WordClouds, distribution des sentiments
- **Prédiction temps réel** : Interface de classification avec scores de confiance et historique  
- **Métriques détaillées** : Comparaison automatisée des modèles, matrice de confusion, courbe ROC
- **Gestion modulaire** : Détection automatique des modèles, sélection et comparaison avec baseline DistilBERT

### Aperçu de l'interface

```
🎯 Dashboard d'Analyse de Sentiment
├── 📊 Analyse Exploratoire
│   ├── Distribution des sentiments (graphiques interactifs)
│   ├── Statistiques textuelles (longueur, fréquence des mots)
│   └── WordClouds différenciés par polarité
├── 🤖 Prédiction en Temps Réel  
│   ├── Saisie libre ou exemples prédéfinis
│   ├── Classification instantanée avec confiance
│   └── Historique des prédictions de session
└── 📈 Métriques Détaillées
    ├── Comparaison multi-modèles vs baseline
    ├── Visualisations (confusion matrix, ROC, historique)
    └── Résumé exécutif adaptatif par modèle
```

#### Page principale
![Page principale du dashboard](images/p09-dashboard-streamlit-mockup-01-main.png)

#### Page de prédiction
![Page de prédiction de sentiment](images/p09-dashboard-streamlit-mockup-03-prediction.png)

### Installation et lancement (local)

Voici les commandes pour une installation avec environnement virtuel :

```bash
# Cloner le repository
git clone <repository-url>
cd oc-ai-engineer-p09-developpez-une-preuve-de-concept

# Installer et configurer Git LFS
sudo apt update && sudo apt install -y git-lfs  # Sur Ubuntu/Debian
# brew install git-lfs                          # Sur macOS

# Configurer Git LFS pour ce repository
git lfs install

# Télécharger les modèles ML volumineux
git lfs pull

# Vérifier que les modèles sont bien téléchargés
ls -lh models/modernbert-sentiment-20250816_1156/model/model.safetensors
# Devrait afficher ~598MB

# Créer l'environnement virtuel
python3 -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate  # Sur Linux/macOS
# venv\Scripts\activate   # Sur Windows

# Installer les dépendances Python
pip install -r requirements.txt

# Lancer l'application
streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false
```

#### 🔍 Vérifications utiles :

```bash
# Vérifier Git LFS
git lfs version

# Voir tous les fichiers trackés par LFS
git lfs ls-files

# Vérifier l'état des fichiers LFS
git lfs status
```

Note importante : Si `git lfs pull` échoue ou que les modèles restent des fichiers pointer (~130 bytes), vous devrez peut-être :
- Vous authentifier sur GitHub avec les bonnes permissions
- Utiliser ``git lfs fetch --all` puis `git lfs checkout`
- Ou télécharger manuellement les modèles depuis l'interface GitHub

#### **🔄 Pour les utilisations suivantes :**

```bash
cd oc-ai-engineer-p09-developpez-une-preuve-de-concept

# Activer l'environnement virtuel
source venv/bin/activate

# Lancer l'application
streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false
```

L'application sera accessible sur `http://localhost:8501`. Le dashboard détecte automatiquement les modèles disponibles dans `models/` et charge leurs métriques pour comparaison.

### GitHub Codespaces + Streamlit + Git LFS

Ce projet est configuré pour fonctionner automatiquement avec **GitHub Codespaces** avec tous les outils nécessaires pour le développement d'applications IA.

#### ⚙️ Configuration automatique

- **🐍 Python 3.11** avec environnement de développement optimisé
- **📦 Git LFS** pour la gestion des modèles de ML volumineux (598MB+)
- **🔥 Streamlit** pour l'interface web interactive
- **🤖 Transformers & PyTorch** pour les modèles d'IA

#### 🎯 Démarrage rapide

1. **Ouvrir dans Codespaces** - L'environnement se configure automatiquement
2. **Lancer l'application** : L'application se lance automatiquement mais vous pouvez la lancer manuellement avec la commande `bash start.sh` (à la racine du projet)
3. **Accéder au dashboard** via le port 8501 (ouverture automatique)

#### 📁 Structure du projet

```
├── .devcontainer/          # Configuration Codespaces
├── main.py            # Point d'entrée principal  
├── models/            # Modèles ML (Git LFS)
├── utils/             # Utilitaires et helpers
├── setup.sh               # Script d'installation
└── start.sh               # Script de lancement
```

#### 🔧 Fonctionnalités

- ✅ **Installation automatique** des dépendances au démarrage
- ✅ **Gestion intelligente** des fichiers volumineux via Git LFS
- ✅ **Vérification automatique** - évite les téléchargements inutiles
- ✅ **Configuration VS Code** optimisée pour Python/IA
- ✅ **Déploiement** prêt pour Streamlit Cloud

> **Note :** Le premier démarrage peut prendre quelques minutes pour télécharger les modèles ML


## 📦 Technologies

- **Langages** : Python  
- **Librairies ML/DL** : PyTorch, Transformers, Scikit-learn  
- **MLOps** : MLFlow, GitHub Actions  
- **Backend/API** : Streamlit  
- **LLM API** : Claude AI  

## A propos 

Projet développé par [David Scanu](https://www.linkedin.com/in/davidscanu14/) dans le cadre du parcours [AI Engineer](https://openclassrooms.com/fr/paths/795-ai-engineer) d'OpenClassrooms :  
*Projet 9 : Développez une preuve de concept*.
