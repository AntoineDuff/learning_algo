import numpy as np
import sys
import load_datasets
import BayesNaif # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés

if __name__ == "__main__":

    """
    C'est le fichier main duquel nous allons tout lancer
    Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
    En gros, vous allez :
    1- Initialiser votre classifieur avec ses paramètres
    2- Charger les datasets
    3- Entrainer votre classifieur
    4- Le tester

    """

    # Initializer vos paramètres

    train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.3)

    # Initializer/instanciez vos classifieurs avec leurs paramètres

    # knn = Knn.Knn(n_neighbors=5)
    # knn.train(train, train_labels)

    # knn.test(test, test_labels)

    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(train, train_labels)

    b = BayesNaif.BayesNaif()
    b.train(train, train_labels)

    b.test(test, test_labels)
    rr = clf.score(test, test_labels)
    # true = []
    # tt = []
    # count = 0
    # b.predict(test[24], test_labels[24])
    # for x, y in zip(test, test_labels):
    #     true.append(clf.predict(np.asarray([x])))
    #     tt.append(b.predict(x, y))
    #     if b.predict(x, y)!= clf.predict(np.asarray([x]))[0]:
    #         print(x, "ERROOOOOOORRR")
    #     print(b.predict(x, y), clf.predict(np.asarray([x]))[0])
    #     count = count +1
    # print(tt)
    # print(true)


    # Charger/lire les datasets

    # Entrainez votre classifieur

    # Tester votre classifieur
