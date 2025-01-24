import numpy as np

# 1. Coder l'algorithme KNN à l'aide de Numpy
def KNN_algo(ensemble_entrainement, labels_entrainement, ensemble_test, k):
    predictions = []
    for donnee_t in ensemble_test:
        distances = []
        for donnee_e in ensemble_entrainement:
            distance = np.sqrt(np.sum((donnee_t - donnee_e) ** 2))
            distances.append(distance)
        indices_k_proches = np.argsort(distances)[:k]
        labels_proches = [labels_entrainement[i] for i in indices_k_proches]
        label_frequent = max(set(labels_proches), key=labels_proches.count) #recherche du max de l'ensemble
        predictions.append(label_frequent)

    return predictions


