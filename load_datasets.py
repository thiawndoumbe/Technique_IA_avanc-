import numpy as np
import pandas as pd
import os

def load_iris_dataset(train_ratio, dataset_path='datasets/bezdekIris.data'):
    """
    Cette fonction a pour but de lire le dataset Iris.

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        dataset_path: chemin vers le fichier contenant le dataset.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels.
    """
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    with open(dataset_path, 'r') as f:
        data = [line.strip().split(',') for line in f if line.strip()]

    np.random.seed(1)
    np.random.shuffle(data)

    num_train = int(len(data) * train_ratio)
    train_data = data[:num_train]
    test_data = data[num_train:]

    train = np.array([list(map(float, row[:-1])) for row in train_data])
    train_labels = np.array([conversion_labels[row[-1]] for row in train_data])
    test = np.array([list(map(float, row[:-1])) for row in test_data])
    test_labels = np.array([conversion_labels[row[-1]] for row in test_data])

    return train, train_labels, test, test_labels


def load_wine_dataset(train_ratio, dataset_path='datasets/binary-winequality-white.csv'):
    """
    Cette fonction a pour but de lire le dataset Binary Wine quality.

    Args:
        train_ratio: le ratio des exemples pour l'entrainement,
        dataset_path: chemin vers le fichier contenant le dataset.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels.
    """
    df = pd.read_csv(dataset_path)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values.astype(int)

    num_train = int(len(data) * train_ratio)
    return data[:num_train], labels[:num_train], data[num_train:], labels[num_train:]


def load_abalone_dataset(train_ratio, dataset_path='datasets/abalone-intervalles.csv'):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles.

    Args:
        train_ratio: le ratio des exemples pour l'entrainement,
        dataset_path: chemin vers le fichier contenant le dataset.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels.
    """
    df = pd.read_csv(dataset_path)

    # Encode non-numeric feature in the first column
    gender_mapping = {'M': 0, 'F': 1, 'I': 2}
    df.iloc[:, 0] = df.iloc[:, 0].map(gender_mapping)

    # Mélanger les données
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    # Séparer les caractéristiques et les étiquettes
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values.astype(int)

    # Séparer les données en ensembles d'entraînement et de test
    num_train = int(len(data) * train_ratio)
    return data[:num_train], labels[:num_train], data[num_train:], labels[num_train:]

