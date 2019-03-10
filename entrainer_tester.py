import numpy as np
import sys
import load_datasets
import BayesNaif # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés

def K_folds(X, y, n_splits=5):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
if __name__ == "__main__":

    # Initializer vos paramètres

    train, train_labels, test, test_labels = load_datasets.load_monks_dataset(1)

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
    # true = []
    # tt = []
    # count = 0
    # for x, y in zip(test, test_labels):
    #     true.append(clf.predict(np.asarray([x])))
    #     tt.append(b.predict(x, y))
    #     if b.predict(x, y)!= clf.predict(np.asarray([x]))[0]:
    #         print(clf.predict_proba(np.asarray([x])))
    #         b.predict(x, y)
    #         print(x, "ERROOOOOOORRR")
    #     print(b.predict(x, y), clf.predict(np.asarray([x]))[0])
    #     count = count +1
    # print(tt)
    # print(true)


    # Charger/lire les datasets

    # Entrainez votre classifieur

    # Tester votre classifieur
