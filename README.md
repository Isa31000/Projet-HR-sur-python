# 🧠 Projet Data - Analyse des performances des employés

## 🔍 Objectif
Ce projet vise à explorer les performances des employés au sein d'une entreprise dans le but d’identifier les écarts significatifs, de détecter les outliers et de proposer des pistes d’amélioration fondées sur des données objectives.

## 🧩 Contexte métier
Une entreprise cherche à mieux comprendre la répartition des performances individuelles, notamment en lien avec des facteurs tels que :
- le nombre d’heures travaillées,
- le niveau d'engagement,
- la satisfaction au travail,
- le rôle et le département.

L’objectif est d’identifier les leviers d’action pour améliorer les performances globales, tout en tenant compte des disparités entre profils.

## 📊 Méthodologie

### 1. 📁 Préparation des données
- Chargement du fichier source
- Nettoyage et traitement des valeurs manquantes
- Normalisation et encodage si nécessaire

### 2. 📈 Statistiques descriptives
- Moyenne, médiane, mode
- Quartiles, variance, écart-type
- Corrélations entre les variables

### 3. 🖼 Visualisation
- Histogrammes, boxplots, pie charts, diagrammes en barres
- Matrices de corrélation
- Grilles multi-axes pour croiser plusieurs dimensions

### 4. 🚨 Détection des outliers
- Méthode IQR (1.5 * IQR Rule)
- Analyse visuelle via boxplots et scatterplots
- Mise en évidence des cas extrêmes

## 🧾 Résultat
Le notebook produit plusieurs visualisations et un rapport graphique complet, exporté au format PDF et PNG. Ces supports facilitent l’interprétation des données par des profils non techniques.

## 📁 Fichiers

- [`nettoyage_donnees.ipynb`](nettoyage_donnees.ipynb) : Préparation des données
- [`data_processing.ipynb`](data_processing.ipynb) : Code source du projet
- [`dashboard.ipynb`](dashboard.ipynb) : Visualisation du rapport
- [`HRDataset.csv`](HRDataset.csv) : Données brutes
- [`palette.png`](palette.png) : Palette de couleurs utilisée
- [`figures/`](figures/) : Graphiques exportés (PNG et PDF)
- [`figures/all_figures.pdf`](figures/all_figures.pdf) : Rapport visuel complet
- [`README.md`](README.md) : Présentation du projet


## ⚙️ Technologies utilisées
- Python (Pandas, Seaborn, Matplotlib, NumPy)
- Jupyter Notebook
- Visualisation avancée via `matplotlib.gridspec`

## 💡 Pour aller plus loin
Ce projet pourrait être étendu par :
- Une modélisation prédictive (classification des performances)
- Une analyse multivariée (ACP ou clustering)
- L’intégration de données RH externes (ancienneté, formation, etc.)

---

🎯 *Ce projet montre ma capacité à combiner rigueur analytique, sens métier et communication visuelle efficace.* N’hésitez pas à explorer le notebook et à me contacter pour en discuter !
