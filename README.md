# 📝 Projet 9 – Développez une preuve de concept : Amélioration d’un modèle d’analyse de sentiment de tweets

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

---

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
- `claude-3-5-haiku-20241022` (avec batch processing)

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
| **Claude 3.5 Haiku 20241022**                   | LLM           | Zero-shot (API)    | Meilleure compréhension contextuelle, batch processing                       | Coût plus élevé, temps de réponse plus long                   |
| **ModernBERT (fine-tuning)**                    | Transformer   | Transfer learning  | Pré-entraînement massif (2T tokens), optimisé mémoire & vitesse, séquence longue | Nécessite entraînement, tuning hyperparamètres                |

> **Note** : les performances réelles seront mesurées sur le même jeu de test Sentiment140 afin d’assurer une comparaison équitable.

---

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

---

## 📚 Références

1. Go, A., Bhayani, R., & Huang, L. (2009). [Twitter Sentiment Classification using Distant Supervision](http://help.sentiment140.com/for-students)  
2. Wolf, T. et al. (2020). [Transformers: State-of-the-Art Natural Language Processing](https://arxiv.org/abs/1910.03771)  
3. Answer.AI. (2024). [ModernBERT: Efficient Transformer for Long-Sequence NLP](https://huggingface.co/answerdotai/ModernBERT-base)

---

## 📦 Technologies

- **Langages** : Python  
- **Librairies ML/DL** : PyTorch, Transformers, Scikit-learn  
- **MLOps** : MLFlow, GitHub Actions  
- **Backend/API** : Streamlit  
- **LLM API** : Claude AI  

