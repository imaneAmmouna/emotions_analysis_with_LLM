# analyse automatique des émotions humaine à partie de texte avec LLM

## Projet 1 : Classification binaire des sentiments (Amazon Reviews)

### Description
Ce projet consiste à construire un modèle de classification binaire des sentiments à partir d'avis clients (Amazon Reviews).  
Il inclut le prétraitement des données, l'extraction automatique des aspects, l'entraînement d'un modèle BERT et l'évaluation des performances.


### Dataset
- **Nom** : Amazon Reviews  
- **Origine** : Dataset public disponible sur Kaggle  
- **Contenu** : Fichiers texte compressés (.bz2) contenant des millions d’avis annotés avec des labels binaires  
- **Préparation** : Décompression, transformation en tableau Pandas, échantillonnage de 40 000 avis pour accélérer l’entraînement.

### Méthodologie
- Prétraitement des textes : nettoyage, tokenisation, padding
- Extraction des aspects via spaCy
- Entraînement du modèle BERT (`bert-base-uncased`)
- Évaluation : accuracy, precision, recall, f1
- Prédiction sur de nouveaux textes

### Prétraitement des données
1. Nettoyage du texte : suppression de caractères inutiles et conversion en minuscules  
2. Tokenisation et padding : les textes sont convertis en séquences de tokens avec une longueur maximale de 512  
3. Encodage des labels : transformation des labels en 0 (positif) ou 1 (négatif)  
4. Équilibrage des classes pour éviter le biais du modèle  

### Extraction automatique des aspects
- Utilisation de **spaCy** pour extraire les **noms communs et propres** présents dans chaque avis  
- Chaque phrase reçoit une liste d’aspects (éléments du produit ou thèmes évoqués)  
- Si aucun aspect n’est détecté, le modèle assigne l’aspect `General`

### Modèle
- **BERT (`bert-base-uncased`)** adapté à la classification binaire  
- Entraînement avec `transformers.Trainer` et PyTorch  
- Paramètres principaux :
  - Epochs : 1 (pour démonstration, peut être augmenté)
  - Batch size : 8  
  - Optimiseur : par défaut HuggingFace AdamW

### Évaluation
- Split du dataset : 80% entraînement, 20% test  
- Metrics utilisées :
  - Accuracy (précision globale)
  - Precision (précision sur la classe positive)
  - Recall (rappel sur la classe positive)
  - F1-score (moyenne harmonique de precision et recall)  
- Visualisation des résultats sous forme de graphique à barres

### Prédiction sur de nouveaux textes
- Le modèle peut analyser de nouvelles phrases ou avis
- Extraction automatique des aspects pour chaque phrase
- Classification du sentiment (Positive / Negative) avec le modèle entraîné
- Exemple :
```python
predictions = classify_text(
    "The packaging was good, but the product is bad.", 
    model, 
    tokenizer
)
```
---
## Projet 2 : Classification multi-classe des emotions (GoEmotions)

### Description
Ce projet consiste à construire un **modèle de classification multi-classe d'émotions** à partir de textes en utilisant le dataset [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions).  

Deux modèles ont été explorés :  
- **GRU (Gated Recurrent Unit)** : modèle RNN bidirectionnel avec embeddings et Dropout.  
- **BERT (Fine-tuning)** : modèle pré-entraîné `bert-base-uncased` adapté à la classification des émotions.

Le projet inclut la préparation des données, l’entraînement des modèles et l’évaluation de leurs performances.

### Objectifs
- Comprendre l'analyse automatique des émotions dans le NLP.
- Comparer les performances de GRU et BERT sur un même dataset.
- Fournir un rapport LaTeX détaillé.

### Dataset
- **Nom** : GoEmotions
- **Origine** : Google Research
- **Taille** : ~58 000 phrases annotées
- **Nombre de classes** : 27 émotions + neutral  
- **Émotions sélectionnées pour le projet** : `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`.

### Méthodologie
1. **Prétraitement des données**  
   - Nettoyage du texte (suppression de caractères inutiles, mise en minuscule)  
   - Tokenisation et padding des séquences  
   - Encodage des labels  

2. **Modèles utilisés**  
   - **GRU** : Embedding → SpatialDropout → Bidirectional GRU → Dense + Dropout → Softmax  
   - **BERT** : Fine-tuning du modèle pré-entraîné `bert-base-uncased` avec couche Dense pour classification  

3. **Entraînement et évaluation**  
   - Séparation train/validation/test  
   - Metrics : Accuracy, Loss, Matrice de confusion  
   - Comparaison de la vitesse et de la performance entre GRU et BERT

---
## Explication des modèles

### 1️⃣ Modèle BERT (Bidirectional Encoder Representations from Transformers)
BERT est un **modèle pré-entraîné basé sur l’architecture Transformer** qui permet de capturer le contexte bidirectionnel des mots dans une phrase.

- **Principe** : Contrairement aux modèles classiques, BERT lit le texte **dans les deux sens simultanément**, ce qui permet de mieux comprendre le sens des mots selon leur contexte.
- **Pré-entraînement** : BERT a été pré-entraîné sur de grandes quantités de texte pour apprendre des représentations riches des mots (Masked Language Modeling et Next Sentence Prediction).
- **Adaptation au projet** :  
  - Nous avons utilisé `bert-base-uncased` et ajouté une **couche dense de classification** au-dessus pour prédire les émotions ou le sentiment.  
  - Le modèle prend des tokens en entrée, génère des embeddings contextuels, puis la couche dense produit une probabilité pour chaque classe.
- **Avantages** : 
  - Capture le contexte des mots très précisément  
  - Performant même avec peu de données annotées grâce au pré-entraînement  
- **Inconvénients** : 
  - Très coûteux en mémoire et en calcul  
  - Temps d’entraînement plus long que les modèles RNN simples
 
### 2️⃣ Modèle GRU (Gated Recurrent Unit)
GRU est un **type de réseau de neurones récurrent (RNN)** conçu pour traiter des séquences, comme les phrases ou documents.

- **Principe** : GRU garde en mémoire les informations importantes dans une séquence et peut oublier celles qui sont moins pertinentes grâce à ses **portes d’entrée et de réinitialisation**.
- **Architecture utilisée** : 
  - **Embedding Layer** : transforme chaque mot en vecteur dense  
  - **SpatialDropout** : régularisation pour éviter l’overfitting  
  - **Bidirectional GRU** : lit la séquence dans les deux sens (avant et arrière) pour capturer le contexte complet  
  - **Dense + Dropout** : couche fully-connected pour transformer les features extraites en vecteurs de sortie, avec régularisation  
  - **Softmax** : génère une probabilité pour chaque classe d’émotion
- **Avantages** : 
  - Moins coûteux en mémoire que BERT  
  - Adapté aux séquences de texte, capable de capturer l’ordre des mots
- **Inconvénients** : 
  - Moins performant que BERT sur des textes complexes ou longs  
  - Ne bénéficie pas du pré-entraînement massif sur de grands corpus

