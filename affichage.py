import numpy 
import load_datasets
from matplotlib import pyplot
import BayesNaif
import Knn

data, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.8)
# pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
pairs = [(0, 1)]
for (f1, f2) in pairs:

    X = numpy.array([[i[f1],i[f2]] for i in data])
    y = numpy.array(train_labels)

    classifieurs = [
        BayesNaif.BayesNaif(),
        Knn.Knn()
    ]
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, step=0.05), numpy.arange(y_min, y_max, step=0.05))
    colors = ['blue', 'green','red']
    pyplot.jet()
    # On crée une figure à plusieurs sous-graphes
    fig, subfigs = pyplot.subplots(1, 2, sharex='all', sharey='all')
    # _times.append(time.time())
    # print(total_data['feature_names'][f2], "en fonction de :", total_data['feature_names'][f1])
    for clf,subfig in zip(classifieurs, subfigs.reshape(-1)):
        # TODO Q2B
        # Entraînez le classifieur
        clf.train(X,y)
        # TODO Q2B
        # Obtenez et affichez son erreur (1 - accuracy)
        # Stockez la valeur de cette erreur dans la variable err

        # err = 1 - clf.score(X,y)
        # print("Erreur pour",clf.__class__.__name__," :",err )

        # TODO Q2B
        # Utilisez la grille que vous avez créée plus haut
        # pour afficher les régions de décision, de même
        # que les points colorés selon leur vraie classe

        k = numpy.c_[xx.ravel(), yy.ravel()]
        ee = numpy.arange(len(k))
        Z = numpy.array(list(map(lambda x, label: clf.predict(x, 1), k, ee)))
        Z = Z.reshape(xx.shape)

        subfig.contourf(xx, yy, Z.reshape(xx.shape),
            alpha = 0.75)
        for i in range(3):
            x_val = []
            y_val = []
            for j,k in enumerate(y):
                if k==i:
                    x_val.append(X[:, 0][j])
                    y_val.append(X[:, 1][j])
            subfig.scatter(x_val, y_val, c=colors[i], edgecolors='k')#, label=total_data['target_names'][i])
        

        # Identification des axes et des méthodes

        subfig.set_xlabel('longueur du sépale (en cm)')
        subfig.set_ylabel('largeur du sépale (en cm)')
        subfig.set_title(clf.__class__.__name__, fontsize = 10)
        subfig.tick_params(labelbottom=True,labelleft=True)
    pyplot.show()