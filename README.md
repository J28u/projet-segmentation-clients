# OC/DS Projet 5 : Segmentez des clients d'un site e-commerce
Formation OpenClassrooms - Parcours data scientist - Projet Professionnalisant (Février - Mars 2023)

## Secteur : 
Communication et marketing ; Commerce et distribution

## Technologies utilisées : 
  * Jupyter Notebook
  * Python

## Mots-clés : 
classification non supervisée, clustering, ACP

## Le contexte : 
Les équipes marketing du site e-commerce brésilien Olist souhaitent optimiser leurs campagnes de communication. Pour cela elles ont besoin d’identifier les différents profils de consommateurs qui achètent en passant par le site.

## La mission : 
Proposer un modèle permettant de regrouper les clients au comportement similaire en segments facilement utilisables par le service marketing.

Établir un contrat de maintenance estimant la fréquence à laquelle le modèle doit être mis à jour, i.e. ré-entrainé, pour que la segmentation obtenue reste pertinente.

 ## Livrables :
 * notebook_exploration.ipynb : notebook de l'analyse exploratoire
 * notebook_modelisation.ipynb : notebook d’essais des différentes approches de modélisation
 * notebook_maintenance.ipynb : notebook de simulation pour déterminer la fréquence nécessaire de mise à jour du modèle de segmentation
 * toolbox.py : fonctions utilisées dans le notebook
 * presentation.pdf : support de présentation pour la soutenance

## Algorithme retenu : 
KMeans

## Méthodologie suivie : 
1. Nettoyage du jeu de données :
* aggrégation des différentes tables
* imputation valeurs manquantes
* traitement des valeurs extrêmes

2. Traitement des données :
* feature engineering : variables RFM
* transformation des variables (StandardScaler, Passage au logarithme)

3. Modélisation :
* test différents algo et combinaisons de variables :
	- DBSCAN
	- Classification Hiérarchique Ascendante
	- Kmeans
* choix du meilleur modèle selon 3 critères :
	- les clusters ont du sens quand on les compare aux connaissances spécifiques au domaine (boxplots, radarplots)
	- les clusters sont équilibrés (diagramme circulaire)
    - les clusters sont homogènes et séparés (coefficient de silhouette)

4. Simulation pour déterminer la fréquence nécessaire de mise à jour du modèle de segmentation :
* à chaque intervalle de temps ti on compare les clusterings de Mi sur Ci et de M0 sur Ci, en calculant l'indice de Rand ajusté. 
* le modèle devra être réentrainé dès que l'indice de Rand Ajusté passe en dessous de 0.8 


## Compétences acquises :  
* Adapter les hyperparamètres d’un algorithme non supervisé afin de l’améliorer
* Évaluer les performances d’un modèle d’apprentissage non supervisé
* Transformer les variables pertinentes d’un modèle d’apprentissage non supervisé
* Mettre en place le modèle d’apprentissage non supervisé adapté au problème métier

## Data source : 
https://www.kaggle.com/olistbr/brazilian-ecommerce
