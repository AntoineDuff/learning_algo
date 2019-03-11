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
        # determine the minimum splittable len of the dataset for n folds 
        stop = len(X) - len(X) % self.n_splits
        fold_len = int(stop / self.n_splits)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        #reduce the list lenght to fit an even split
        indices = indices[:stop]
        
        train_ind_list = []
        test_ind_list = []

        for i in range(self.n_splits):
            train_ind_list.append(indices[:-fold_len])
            test_ind_list.append(indices[-fold_len:])

            indices = np.roll(indices, fold_len)

        return train_ind_list, test_ind_list

if __name__ == "__main__":

    # Initializer vos paramètres

    train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.8)

    # Initializer/instanciez vos classifieurs avec leurs paramètres

    knn_clf = Knn.Knn(n_neighbors=5)

    kf = K_folds(n_splits=5)
    train_kf, train_label_kf = kf.split(train, train_labels)

    avg_score = 0
    for train_inds, test_inds in zip (train_kf, train_label_kf):
        X_train = train[train_inds]
        y_train = train_labels[train_inds]
        X_test = train[test_inds]
        y_test = train_labels[test_inds]

        knn_clf.train(X_train, y_train)
        avg_score += knn_clf.test(X_test, y_test, muted=True)

    avg_score = avg_score / kf.n_splits
    print(avg_score)

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
