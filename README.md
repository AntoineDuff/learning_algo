# Implementation of Knn and Naive Bayes classfiers
# Classe Knn:
Knn est une classe implémentant un modèle de classement suivant l'algorithme des k plus proches voisins.
- Le contructeur __init__() de la classe contient prend en paramètre le métrique à utiliser pour le calcul des distances ainsi que le nombre de voisins à utiliser. 
- train() prend les données ainsi que leur étiquette pour l'entrainement du modèle et réalise l'entrainement de ce dernier.
- predict() donne la prédiction du modèle sur une donnée quelconque.
- accuracy() retourne le score du modèle.
- confusion_matrix() prend les données test ainsi que leur étiquette en argument et retourne la matrice de confusion générée selon les données à prédires.
- recall() prend la matrice de confusion en argument et retourne le rappel selon la classe.
- precision() prend la matrice de confusion en argument et retourne la précision selon la classe.
- test() prend les données test ainsi que leur étiquette en argument et retourne le score du modèle. Il est aussi possible d'afficher le rappel, la matrice de confusion ainsi que la précision en mettant l'argument muted à True.

# Classe BayesNaïf
BayesNaif est une classe suivant l'implémentation classique du modèle de classement bayes naïf.
- Le contructeur par défaut est utilisé pour cette classe sans arguments.
- train() prend les données ainsi que leur étiquette pour l'entrainement du modèle et réalise l'entrainement de ce dernier.
- predict() donne la prédiction du modèle sur une donnée quelconque.
- accuracy() retourne le score du modèle.
- confusion_matrix() prend les données test ainsi que leur étiquette en argument et retourne la matrice de confusion générée selon les données à prédires.
- recall() prend la matrice de confusion en argument et retourne le rappel selon la classe.
- precision() prend la matrice de confusion en argument et retourne la précision selon la classe.
- test() prend les données test ainsi que leur étiquette en argument et retourne le score du modèle. Il est aussi possible d'afficher le rappel, la matrice de confusion ainsi que la précision en mettant l'argument muted à True.

# Classe K_folds
La classe K_folds implémente la méthode de génération de données en k-plis pour la validation croisée.
- Le constructeur __init__() prend en argument le nombre de plis à utiliser.
- split() prend en arguement les données à diviser ainsi que leur étiquette. Cette méthode retourne une liste de listes des indices des données et des étiquettes à utiliser dans le jeu de donnée initiale pour chacun des plis.


# Division du travail
L'ensemble de l'implémentation des classes à été réaliser par les deux membres de l'équipe Louis-Gabriel Maltais et Antoine Dufour. Aucune difficulté particulière à été rencontré lors de l'implémentation du code.