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

class BayesNaif:
        
    def train(self, train, train_labels):
        self.cnt_classes_occ = np.bincount(train_labels)
        self._means = self._compute_classes_means(train, train_labels)
        self._priors = self._compute_classes_priors(train_labels)
        self._cov_mats = self._compute_classes_cov_mat(train, train_labels)
        self._shared_cov_mat = self._shared_naive_cov_mats()

    def predict(self, exemple, label):
        h = ((-1/2) * np.sum(np.power(exemple[None, :] - self._means, 2) / 
            np.diagonal(self._shared_cov_mat)[None, :], axis=1) +
            np.log(self._priors))
        pred = np.argmax(h)

        return pred

    def accuracy(self, pred_label, true_label):

        Y = [pred_label[j] for j in range(len(pred_label)) if pred_label[j]==true_label[j] ]
        score = len(Y)/len(pred_label)

        return score

    def test(self, test, test_labels):
        pred = list(map(lambda x, label: self.predict(x, label), test, test_labels))

        score = self.accuracy(pred, test_labels)
        print("Accuracy: ", score, '\n')

    def _compute_classes_means(self, train, train_labels):
        classes = np.unique(train_labels)
        cnt = self.cnt_classes_occ
        means = np.zeros(shape=(len(classes), train.shape[1]))
        np.add.at(means, train_labels, train)
        means /= cnt[:, None]

        return means

    def _compute_classes_priors(self, train_labels):
        priors = self.cnt_classes_occ / float(len(train_labels))

        return priors

    def _compute_classes_cov_mat(self, train, train_labels):
        classes = np.unique(train_labels)

        cov_mat_i = []
        for classe in classes:
            mat_i = []
            for j, data in enumerate(train):
                if(classe == train_labels[j]):
                    mat_i.append(data)

            mat_i = np.asarray(mat_i)
            cov_mat_i.append(np.cov(mat_i.T, bias=True))

        return np.asarray(cov_mat_i)

    def _shared_naive_cov_mats(self):
        S_shared = np.sum(self._cov_mats * 
                   np.identity(len(self._means[0]))[None, :] *
                   self._priors[:, None, None], axis=0)

        return S_shared
        



                