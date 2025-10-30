# Checkpoint Machine Learning Prediction

L’objectif de ce projet est de predire si une demande de prêt sera **approuvee** ou **refusee**, a partir d’informations de base sur les clients
Ce projet fait partie du checkpoint en **Machine Learning avec Streamlit**

J’ai choisi ce projet parce qu’il représente un cas concret et simple : les banques prennent chaque jour ce genre de décision  
C’est un bon moyen de pratiquer la classification supervisee et de créer une application interactive

## Données
Le dataset utilisé s’appelle **Simple Loan Classification Dataset**, depuis **Kaggle**.

Chaque ligne correspond a un client, avec les colonnes suivantes :
- `age` : age du client  
- `gender` : sexe  
- `occupation` : profession  
- `education_level` : niveau d education  
- `marital_status` : statut marital  
- `income` : revenu  
- `credit_score` : score de credit  
- `loan_status` : variable cible (approuvé ou refusé)


## Preparation et nettoyage
Avant l’entraînement :
1. Suppression des doublons  
2. Traitement des valeurs manquantes  
3. Encodage des variables catégorielles avec **OneHotEncoder**  
4. Division du dataset :  
   - 80 % pour l’entraînement  
   - 20 % pour le test  
5. Normalisation automatique via le `ColumnTransformer` integré dans le pipeline


## Selection du modele
J’ai choisi *la regression logistique* pour plusieurs raisons :
- Le problme est **binaire** (0 = refus, 1 = approuvé)
- Le dataset est petit
- La régression logistique est **rapide a entrainer**, interpretable, et donne des probabilités
- Elle gere bien les variables encodees avec OneHotEncoder

J’ai testé aussi un arbre de decision, mais il surapprenait rapidement  
La régression logistique donnait de meilleures metriques globales et plus de stabilité

## Evaluation du modele
Après entrainement, le modele a été evalué avec plusieurs métriques :

- **Accuracy (Precision)** : 0.92  
- **F1-score** : 0.94  
- **ROC-AUC** : 1.00  

### Rapport de classification
| Classe | Precision | Rappel | F1-score |
| 0 (Refusé) | 0.75 | 1.00 | 0.85 |
| 1 (Approuvé) | 1.00 | 0.90 | 0.94 |

J’ai ajouté une validation croisée pour vérifier la stabilité des scores  
Résultat moyen du F1-score sur 5 folds : **0.91 ± 0.04**

Ces chiffres restent à prendre avec prudence, car le jeu de données est petit et déséquilibré.

## Interpretation
- Le **revenu** et le **score de credit** sont les variables les plus influentes  
- Les profils jeunes avec un bon score de crédit ont plus de chances d’etre approuvés  
- Le modèle reste basique mais fonctionne bien pour une démonstration

---

## Application Streamlit
Une application Streamlit a été développée pour permettre de faire des prédictions en direct

### Fonctionnalités :
- Formulaire intuitif avec sliders et listes déroulantes  
- Affichage du résultat clair : ✅ Approuvé / ❌ Refusé  

