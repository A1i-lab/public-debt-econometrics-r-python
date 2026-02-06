# Documentation technique

## Architecture de l'application

L'application est développée avec Gradio et s'articule autour de cinq modules principaux :

### 1. Module de chargement des données

Restructuration du fichier Excel au format large vers un format long adapté aux analyses temporelles.

Variables extraites :
- Dette publique (% du PIB)
- Croissance du PIB
- Taux d'inflation
- Dépenses et recettes publiques
- Indicateurs structurels (chômage, démographie, etc.)

### 2. Module de modélisation

Algorithme : Random Forest Regressor

Paramètres retenus :
- 200 arbres de décision
- Profondeur maximale : 15
- Validation croisée : 5 plis
- Normalisation : StandardScaler

### 3. Module de visualisation

Génération de graphiques interactifs avec Plotly :
- Séries temporelles historiques
- Projections futures
- Analyses comparatives
- Clustering multidimensionnel

### 4. Module de projection

Méthodologie :
- Calcul des moyennes mobiles (3 dernières années)
- Application du modèle entraîné
- Génération de scénarios 2025-2030

### 5. Module d'export

Génération de rapports Excel structurés par pays avec :
- Données historiques complètes
- Projections détaillées
- Métadonnées et sources

## Workflow de traitement

Excel brut → Restructuration → Nettoyage → Normalisation → Entraînement ML → Interface Gradio


## Performance du modèle

Métriques obtenues :
- R² : environ 0.85-0.90 selon validation croisée
- MAE : variable selon les pays
- RMSE : mesure la précision des projections

## Dépendances critiques

- Python 3.13
- Gradio pour l'interface utilisateur
- Scikit-learn pour les algorithmes ML
- Pandas pour la manipulation de données
- Plotly pour les visualisations interactives

## Déploiement

L'application est hébergée sur Hugging Face Spaces, offrant :
- Disponibilité 24/7
- URL permanente
- Pas de gestion serveur requise
- Intégration continue depuis GitHub

## Limites techniques

- Les projections supposent la stabilité des relations passées
- Pas de prise en compte des chocs exogènes
- Qualité dépendante de la complétude des données historiques
- Pas de modélisation des interactions entre pays

## Évolutions possibles

- Intégration d'un modèle dynamique avec variables retardées
- Ajout d'intervalles de confiance pour les projections
- Prise en compte d'effets fixes temporels
- API REST pour intégration externe
