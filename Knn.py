"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 methodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement
	* predict 	: pour prédire la classe d'un exemple donné
	* test 		: pour tester sur l'ensemble de test
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les méthodes test, predict et test de votre code.
"""

import numpy as np

class Knn:
    def __init__(self, n_neighbors=5, metric='minkowski'):
        self._n_neighbors = n_neighbors
        self._metric = metric

    def train(self, train, train_labels, muted=False):
        self._data = train.copy()
        self._labels = train_labels.copy()

        #test on training data
        if not muted:
            print('Test on training data:')
            self.test(train, train_labels, muted=muted)

    def predict(self, exemple, label):
        #eucledian distance
        if self._metric == 'euclidean':
            dist = list(map(lambda x : np.linalg.norm(exemple - x), self._data))
        
        if self._metric == 'minkowski':
            def minkowski_dist(vec):
                p = vec.shape[0]
                return np.sum(np.absolute(vec) ** p) ** (1 / p)

            dist = list(map(lambda x : minkowski_dist(exemple - x), self._data))
        
        if self._metric == 'manhattan':
            dist = list(map(lambda x : np.sum(np.absolute(exemple - x)), self._data))

        sorted_data, sorted_labels = zip(*sorted(zip(dist, list(range(len(self._labels))))))
        predicted_labels = np.take(self._labels, sorted_labels[:self._n_neighbors])
        prediction = np.argmax(np.bincount(predicted_labels))

        return prediction

    def accuracy(self, pred_label, true_label):

        Y = [pred_label[j] for j in range(len(pred_label)) if pred_label[j]==true_label[j] ]
        score = len(Y)/len(pred_label)

        return score

    def confusion_matrix(self, pred_label, true_label):
        labels = np.unique(true_label, return_counts=True)

        #build the confusion matrix
        #row = prediction col=actual class
        confusion_mat = np.zeros((len(labels[0]), len(labels[0])))
        
        for pred_y, true_y in zip(pred_label, true_label):
            confusion_mat[pred_y][true_y] += 1

        return confusion_mat

    def recall(self, confusion_mat):
        recall =  np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=0)

        return recall

    def precision(self, confusion_mat):
        precision =  np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=1)

        return precision

    def test(self, test, test_labels, muted=False):
        #muted: print or do not print results
        
        pred = list(map(lambda x, label: self.predict(x, label), test, test_labels))

        score = self.accuracy(pred, test_labels)
        confusion_matrix = self.confusion_matrix(pred, test_labels)
        precision = self.precision(confusion_matrix)
        recall = self.recall(confusion_matrix)
        if not muted:
            print("Accuracy: ", score, '\n')
            print("Confusion Matrix:\n", confusion_matrix, '\n')
            print("Precision: ", precision, '\n')
            print("Recall: ", recall, '\n')

        return score



            