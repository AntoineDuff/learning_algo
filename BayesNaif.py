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
        
    def train(self, train, train_labels, muted=False):
        self.cnt_classes_occ = np.bincount(train_labels)
        self._means = self._compute_classes_means(train, train_labels)
        self._priors = self._compute_classes_priors(train_labels)
        self._cov_mats, self.diag_cov_mat = self._compute_classes_cov_mat(train, train_labels)
        self._shared_cov_mat = self._shared_naive_cov_mats()

        if not muted:
            print('Test on training data:')
            self.test(train, train_labels, muted=muted)

    def predict(self, exemple, label):
        # h = ((-1/2) * np.sum(np.power(exemple[None, :] - self._means, 2) / np.diagonal(self._shared_cov_mat)[None, :], axis=1) + np.log(self._priors))
        h = self.cpu_log_likelihood(np.asarray(exemple))
        # pred = np.argmax(h)
        pred = np.argmax(h)

        return pred

    def accuracy(self, pred_label, true_label):

        Y = [pred_label[j] for j in range(len(pred_label)) if pred_label[j]==true_label[j] ]
        score = len(Y)/len(pred_label)

        return score

    def confusion_matrix(self, pred_label, true_label):
        labels = np.unique(true_label, return_counts=True)

        #build the confusion matrix
        #y = prediction x=actual class
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
            print("Recall: ",recall, '\n')

        return score


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
        diag_cov_mat = []
        for classe in classes:
            mat_i = []
            for j, data in enumerate(train):
                if(classe == train_labels[j]):
                    mat_i.append(data)

            mat_i = np.asarray(mat_i)
            cov_mat_i.append(np.cov(mat_i.T, bias=True))
            diag_cov_mat.append(np.diagonal(np.cov(mat_i.T, bias=True)))

        return np.asarray(cov_mat_i), np.asarray(diag_cov_mat)


    def _shared_naive_cov_mats(self):
        S_shared = np.sum(self._cov_mats * 
                   np.identity(len(self._means[0]))[None, :] *
                   self._priors[:, None, None], axis=0)

        return S_shared

    def cpu_log_likelihood(self, X):
        log_likelihood = []
        for i in range(np.size(self.cnt_classes_occ)):
            l_val = np.log(self._priors[i])
            temp = - 0.5 * np.sum(np.log(2. * np.pi * self.diag_cov_mat[i, :]))
            temp -= 0.5 * np.sum(((X - self._means[i, :]) ** 2) / (self.diag_cov_mat[i, :]))
            log_likelihood.append(l_val + temp)

        log_likelihood = np.transpose(np.array(log_likelihood))
        
        return log_likelihood
        



                