#MARIUS CASAMIAN L3IA
#cm205773, marius.casamian@etu.unice.fr
#TP1: Python, Pandas, Numpy, KNN

import pandas as pd
import numpy as np
from KNN import KNN_algo

if __name__ == '__main__':
    # Charger les données
    data = pd.read_csv("data/titanic/train.csv")
    print("Succès : Lecture des données")
    # 1. Taille du jeu de données
    print(f"Taille du jeu de données : {data.shape}")

    # 2. Colonnes
    print(f"Noms des colonnes : {data.columns.tolist()}")

    # 3. Dix premières lignes
    print(f"10 premières lignes :\n{data.head(10)}")

    # 4. Tarif le plus élevé
    print(f"Tarif le plus élevé : {data['Fare'].max()}")

    # 5. Passagers ayant plus de 60 ans
    print(f"Nombre de passagers ayant plus de 60 ans : {data[data['Age'] > 60].shape[0]}")

    # 6. Passagers entre 30 et 40 ans
    print(f"Nombre de passagers ayant entre 30 et 40 ans : {data[(data['Age'] >= 30) & (data['Age'] <= 40)].shape[0]}")

    # 7. Pourcentage de survie
    print(f"Pourcentage de personnes ayant survécu : {data['Survived'].mean() * 100:.2f}%")

    # 8. Pourcentage de survie par sexe
    print(f"Pourcentage de survie des femmes : {data[data['Sex'] == 'female']['Survived'].mean() * 100:.2f}%")
    print(f"Pourcentage de survie des hommes : {data[data['Sex'] == 'male']['Survived'].mean() * 100:.2f}%")

    # 9. Taux de survie par classe
    print(f"Taux de survie par classe :\n{data.groupby('Pclass')['Survived'].mean() * 100}")

    # Partie 2 : Prétraitement car impossible de faire la correlation sans erreurs
    # 1. Encodage
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # 2. Imputation NA par moyenne
    data['Age'] = data['Age'].fillna(data['Age'].mean())

    # 3.  Drop des colonnes inutiles
    data = data.drop(columns=['Ticket', 'Name', 'Cabin', 'PassengerId'])

    # 4. X_train (80 % premières lignes sans 'Survived')
    X_train = data.drop('Survived', axis=1).iloc[:int(0.8 * len(data))].to_numpy()

    # 10. Matrice de corrélation
    print(f"Matrice de corrélation :\n{data.corr()}")

    # 5. y_train (80 % premières lignes de 'Survived')
    y_train = data['Survived'].iloc[:int(0.8 * len(data))].to_numpy()

    # 6. X_test (20 % dernières lignes sans 'Survived')
    X_test = data.drop('Survived', axis=1).iloc[int(0.8 * len(data)):].to_numpy()

    # 7. y_test (20 % dernières lignes de 'Survived')
    y_test = data['Survived'].iloc[int(0.8 * len(data)):].to_numpy()

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


    # précisiion des predictions
    def calcul_precision(predictions, labels_reels):
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == labels_reels[i]:
                correct += 1
        return (correct / len(predictions)) * 100

    # k=2
    predictions_2 = KNN_algo(X_train, y_train, X_test, k=2)
    precision_2 = calcul_precision(predictions_2, y_test)
    #k=5
    predictions_5 = KNN_algo(X_train, y_train, X_test, k=5)
    precision_5 = calcul_precision(predictions_5, y_test)

    #k=10
    predictions_10 = KNN_algo(X_train, y_train, X_test, k=10)
    precision_10 = calcul_precision(predictions_10, y_test)

    #k=20
    predictions_20 = KNN_algo(X_train, y_train, X_test, k=20)
    precision_20 = calcul_precision(predictions_20, y_test)

    print(f"Les predictions k = 2: {predictions_2} \nLa précision: {precision_2:.2f}%")
    print(f"Les predictions k = 5: {predictions_5} \nLa précision: {precision_5:.2f}%")
    print(f"Les predictions k = 10: {predictions_10} \nLa précision: {precision_10:.2f}%")
    print(f"Les predictions k = 20: {predictions_20} \nLa précision: {precision_20:.2f}%")
    """
    Petite valeur de k (comme k=2), on voit que l'algorithme peut être plus sujet au bruit, ce qui entraîne des erreurs de classification et donc faire baisser la précision.
    Valeur moyenne de k (comme k=5), ici on observe que 'algorithme commence à trouver un bon compromis entre la précision et la capacité à généraliser et offre donc de meilleures prédictions.
    Grande valeur de k (comme k=10 ou k=20),l'algorithme devient plus stable et offre une précision plus constante. 
    Je pense cependant que prendre un k au dessus de 20 donc trop grand doit inclure des voisins qui ne sont pas pertinents, ce qui risque d'introduire des erreurs.
    """
