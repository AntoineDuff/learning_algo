import numpy as np
import sys
import load_datasets
import BayesNaif # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés

class K_folds:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_rand = X[indices]
        y_rand = y[indices]

        # determine the minimum splittable len of the dataset for n folds
        stop = len(X) - len(X) % self.n_splits

        self.splitted_X = np.split(X_rand[:stop], 5)
        self.splitted_y = np.split(y_rand[:stop], 5)



    
if __name__ == "__main__":

    # Initializer vos paramètres

    train, train_labels, test, test_labels = load_datasets.load_monks_dataset(1)


    kf = K_folds(n_splits=5)

    kf.split(train, train_labels)
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
    print(rr)


    # Charger/lire les datasets

    # Entrainez votre classifieur

    # Tester votre classifieur
