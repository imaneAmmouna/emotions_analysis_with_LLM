# analyse automatique des émotions humaine à partie de texte avec LLM

## Description
Ce projet consiste à construire un **modèle de classification multi-classe d'émotions** à partir de textes en utilisant le dataset [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions).  

Deux modèles ont été explorés :  
- **GRU (Gated Recurrent Unit)** : modèle RNN bidirectionnel avec embeddings et Dropout.  
- **BERT (Fine-tuning)** : modèle pré-entraîné `bert-base-uncased` adapté à la classification des émotions.

Le projet inclut la préparation des données, l’entraînement des modèles et l’évaluation de leurs performances.

---

## Objectifs
- Comprendre l'analyse automatique des émotions dans le NLP.
- Comparer les performances de GRU et BERT sur un même dataset.
- Fournir un rapport LaTeX détaillé.

---

## Dataset
- **Nom** : GoEmotions
- **Origine** : Google Research
- **Taille** : ~58 000 phrases annotées
- **Nombre de classes** : 27 émotions + neutral  
- **Émotions sélectionnées pour le projet** : `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`.

---

## Méthodologie
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

