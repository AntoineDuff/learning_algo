import numpy as np
import time
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
        stop = len(X) - (len(X) % self.n_splits)
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

    ###IRIS DATASET###
    # Initializer vos paramètres
    print("IRIS DATASET:")
    # Charger/lire les datasets
    train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.8)

    # Initializer/instanciez vos classifieurs avec leurs paramètres
    kf = K_folds(n_splits=10)
    train_kf, train_label_kf = kf.split(train, train_labels)
    k_neighb = [1, 3, 5, 7, 9, 11]
    metrics = ['minkowski', 'manhattan', 'euclidean']

    best_k = 1
    best_metric = 'minkowski'
    best_score = 0
    for k in k_neighb:
        for metric in metrics:
            knn_clf = Knn.Knn(n_neighbors=k, metric=metric)
            # kf = K_folds(n_splits=10)
            # train_kf, train_label_kf = kf.split(train, train_labels)
            avg_score = 0
            for train_inds, test_inds in zip (train_kf, train_label_kf):
                X_train = train[train_inds]
                y_train = train_labels[train_inds]
                X_test = train[test_inds]
                y_test = train_labels[test_inds]

                knn_clf.train(X_train, y_train, muted=True)
                avg_score += knn_clf.test(X_test, y_test, muted=True)

            avg_score = avg_score / kf.n_splits
            if best_score < avg_score:
                best_score = avg_score
                best_k = k
                best_metric = metric
            # print(f"k = {k}, metric: {metric} ,score: ", avg_score)

    print('Knn classifier:')
    Knn_clf = Knn.Knn(n_neighbors=best_k, metric=best_metric)
    Knn_clf.train(train, train_labels, muted=False)
    start = time.time()
    Knn_clf.test(test, test_labels)
    print('KNN test time: ', time.time() - start)


    print('Naïve Bayes classifier:')
    BN_clf = BayesNaif.BayesNaif()
    BN_clf.train(train, train_labels, muted=False)

    start = time.time()
    BN_clf.test(test, test_labels)
    print('Naïve Bayes test time: ', time.time() - start)

    ###CONGRESSIONNAL DATASET###
    # Initializer vos paramètres
    print("CONGRESSIONNAL DATASET:")
    train, train_labels, test, test_labels = load_datasets.load_congressional_dataset(0.8)

    # Initializer/instanciez vos classifieurs avec leurs paramètres
    kf = K_folds(n_splits=10)
    train_kf, train_label_kf = kf.split(train, train_labels)
    k_neighb = [1, 3, 5, 7, 9, 11]
    metrics = ['minkowski', 'manhattan', 'euclidean']
    best_k = 1
    best_metric = 'minkowski'
    best_score = 0
    for k in k_neighb:
        for metric in metrics:
            knn_clf = Knn.Knn(n_neighbors=k, metric=metric)
            # kf = K_folds(n_splits=10)
            # train_kf, train_label_kf = kf.split(train, train_labels)
            avg_score = 0
            for train_inds, test_inds in zip (train_kf, train_label_kf):
                X_train = train[train_inds]
                y_train = train_labels[train_inds]
                X_test = train[test_inds]
                y_test = train_labels[test_inds]

                knn_clf.train(X_train, y_train, muted=True)
                avg_score += knn_clf.test(X_test, y_test, muted=True)

            avg_score = avg_score / kf.n_splits
            if best_score < avg_score:
                best_score = avg_score
                best_k = k
                best_metric = metric
            # print(f"k = {k}, metric: {metric} ,score: ", avg_score)
    print('Knn classifier:')
    Knn_clf = Knn.Knn(n_neighbors=best_k, metric=best_metric)
    Knn_clf.train(train, train_labels, muted=False)
    start = time.time()
    Knn_clf.test(test, test_labels)
    print('KNN test time: ', time.time() - start)
    print('Naïve Bayes classifier:')
    BN_clf = BayesNaif.BayesNaif()
    BN_clf.train(train, train_labels, muted=False)

    start = time.time()
    BN_clf.test(test, test_labels)
    print('Naïve Bayes test time: ', time.time() - start)

    ###MONK-1 DATASET###
    # Initializer vos paramètres
    print("MONK-1 DATASET:")
    train, train_labels, test, test_labels = load_datasets.load_monks_dataset(1)

    # Initializer/instanciez vos classifieurs avec leurs paramètres
    kf = K_folds(n_splits=10)
    train_kf, train_label_kf = kf.split(train, train_labels)
    k_neighb = [1, 3, 5, 7, 9, 11]
    metrics = ['minkowski', 'manhattan', 'euclidean']
    best_k = 1
    best_metric = 'minkowski'
    best_score = 0
    for k in k_neighb:
        for metric in metrics:
            knn_clf = Knn.Knn(n_neighbors=k, metric=metric)
            # kf = K_folds(n_splits=10)
            # train_kf, train_label_kf = kf.split(train, train_labels)
            avg_score = 0
            for train_inds, test_inds in zip (train_kf, train_label_kf):
                X_train = train[train_inds]
                y_train = train_labels[train_inds]
                X_test = train[test_inds]
                y_test = train_labels[test_inds]

                knn_clf.train(X_train, y_train, muted=True)
                avg_score += knn_clf.test(X_test, y_test, muted=True)

            avg_score = avg_score / kf.n_splits
            if best_score < avg_score:
                best_score = avg_score
                best_k = k
                best_metric = metric
            # print(f"k = {k}, metric: {metric} ,score: ", avg_score)
    print('Knn classifier:')
    Knn_clf = Knn.Knn(n_neighbors=best_k, metric=best_metric)
    Knn_clf.train(train, train_labels, muted=False)
    start = time.time()
    Knn_clf.test(test, test_labels)
    print('KNN test time: ', time.time() - start)

    print('Naïve Bayes classifier:')
    BN_clf = BayesNaif.BayesNaif()
    BN_clf.train(train, train_labels, muted=False)

    start = time.time()
    BN_clf.test(test, test_labels)
    print('Naïve Bayes test time: ', time.time() - start)

    ###MONK-2 DATASET###
    # Initializer vos paramètres
    print("MONK-2 DATASET:")
    train, train_labels, test, test_labels = load_datasets.load_monks_dataset(2)

    # Initializer/instanciez vos classifieurs avec leurs paramètres
    kf = K_folds(n_splits=10)
    train_kf, train_label_kf = kf.split(train, train_labels)
    k_neighb = [1, 3, 5, 7, 9, 11]
    metrics = ['minkowski', 'manhattan', 'euclidean']
    best_k = 1
    best_metric = 'minkowski'
    best_score = 0
    for k in k_neighb:
        for metric in metrics:
            knn_clf = Knn.Knn(n_neighbors=k, metric=metric)
            # kf = K_folds(n_splits=10)
            # train_kf, train_label_kf = kf.split(train, train_labels)
            avg_score = 0
            for train_inds, test_inds in zip (train_kf, train_label_kf):
                X_train = train[train_inds]
                y_train = train_labels[train_inds]
                X_test = train[test_inds]
                y_test = train_labels[test_inds]

                knn_clf.train(X_train, y_train, muted=True)
                avg_score += knn_clf.test(X_test, y_test, muted=True)

            avg_score = avg_score / kf.n_splits
            if best_score < avg_score:
                best_score = avg_score
                best_k = k
                best_metric = metric
            # print(f"k = {k}, metric: {metric} ,score: ", avg_score)
    print('Knn classifier:')
    Knn_clf = Knn.Knn(n_neighbors=best_k, metric=best_metric)
    Knn_clf.train(train, train_labels, muted=False)
    start = time.time()
    Knn_clf.test(test, test_labels)
    print('KNN test time: ', time.time() - start)

    print('Naïve Bayes classifier:')
    BN_clf = BayesNaif.BayesNaif()
    BN_clf.train(train, train_labels, muted=False)

    start = time.time()
    BN_clf.test(test, test_labels)
    print('Naïve Bayes test time: ', time.time() - start)

    ###MONK-3 DATASET###
    # Initializer vos paramètres
    print("MONK-3 DATASET:")
    train, train_labels, test, test_labels = load_datasets.load_monks_dataset(3)

    # Initializer/instanciez vos classifieurs avec leurs paramètres
    kf = K_folds(n_splits=10)
    train_kf, train_label_kf = kf.split(train, train_labels)
    k_neighb = [1, 3, 5, 7, 9, 11]
    metrics = ['minkowski', 'manhattan', 'euclidean']
    best_k = 1
    best_metric = 'minkowski'
    best_score = 0
    for k in k_neighb:
        for metric in metrics:
            knn_clf = Knn.Knn(n_neighbors=k, metric=metric)
            # kf = K_folds(n_splits=10)
            # train_kf, train_label_kf = kf.split(train, train_labels)
            avg_score = 0
            for train_inds, test_inds in zip (train_kf, train_label_kf):
                X_train = train[train_inds]
                y_train = train_labels[train_inds]
                X_test = train[test_inds]
                y_test = train_labels[test_inds]

                knn_clf.train(X_train, y_train, muted=True)
                avg_score += knn_clf.test(X_test, y_test, muted=True)

            avg_score = avg_score / kf.n_splits
            if best_score < avg_score:
                best_score = avg_score
                best_k = k
                best_metric = metric
            # print(f"k = {k}, metric: {metric} ,score: ", avg_score)
    print('Knn classifier:')
    Knn_clf = Knn.Knn(n_neighbors=best_k, metric=best_metric)
    Knn_clf.train(train, train_labels, muted=False)
    start = time.time()
    Knn_clf.test(test, test_labels)
    print('KNN test time: ', time.time() - start)

    print('Naïve Bayes classifier:')
    BN_clf = BayesNaif.BayesNaif()
    BN_clf.train(train, train_labels, muted=False)

    start = time.time()
    BN_clf.test(test, test_labels)
    print('Naïve Bayes test time: ', time.time() - start)