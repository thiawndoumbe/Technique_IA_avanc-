from collections import Counter
import numpy as np

class Knn:
    def __init__(self, k=5, metric="euclidean"):
        """
        Initialisation du classifieur KNN.
        
        Args:
            k: Nombre de voisins à considérer pour la prédiction (int).
            metric: Métrique utilisée pour calculer les distances entre points. 
                    Options : "euclidean" (par défaut), "manhattan" (str).
        """
        self.k = k
        self.metric = metric
        self.train_data = None
        self.train_labels = None

    def train(self, train: np.ndarray, train_labels: np.ndarray) -> None:
        """
        Entraîne le classifieur sur les données d'entraînement.

        Args:
            train: Matrice des exemples d'entraînement (numpy array).
            train_labels: Labels associés aux exemples d'entraînement (numpy array).
        """
        self.train_data = train
        self.train_labels = train_labels

    def _distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calcule la distance entre deux points en fonction de la métrique spécifiée.

        Args:
            x1: Premier point (numpy array).
            x2: Deuxième point (numpy array).
        
        Retours:
            La distance calculée entre x1 et x2 (float).
        
        Lève:
            ValueError: Si la métrique spécifiée n'est pas supportée.
        """
        if self.metric == "euclidean":
            return np.linalg.norm(x1 - x2)
        elif self.metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Métrique non supportée : {self.metric}")

    def predict(self, x: np.ndarray) -> int:
        """
        Prédire la classe d'un exemple donné.

        Args:
            x: Exemple à prédire (numpy array).

        Retours:
            Classe prédite pour l'exemple x (int).
        """
        distances = np.array([self._distance(x, train_sample) for train_sample in self.train_data])
        k_indices = np.argsort(distances)[:self.k]  # Indices des k voisins les plus proches
        k_nearest_labels = self.train_labels[k_indices]  # Labels des k voisins
        return Counter(k_nearest_labels).most_common(1)[0][0]  # Classe majoritaire parmi les voisins

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Évalue les performances du classifieur sur un ensemble de test.

        Args:
            X: Matrice des exemples de test (numpy array).
            y: Labels associés aux exemples de test (numpy array).

        Retours:
            accuracy: Taux de classification correcte (float).
            precision: Moyenne de la précision par classe (float).
            recall: Moyenne du rappel par classe (float).
            f1: Moyenne du F1-score par classe (float).
            confusion_matrix: Matrice de confusion représentant les performances (numpy array).
        """
        # Prédictions pour tous les exemples de test
        predictions = np.array([self.predict(x) for x in X])
        accuracy = np.mean(predictions == y)

        # Calcul de la matrice de confusion
        unique_labels = np.unique(np.concatenate((y, predictions)))
        confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        for true_label, pred_label in zip(y, predictions):
            confusion_matrix[label_to_index[true_label], label_to_index[pred_label]] += 1

        # Calcul des métriques : précision, rappel, et F1-score
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0, where=confusion_matrix.sum(axis=0) != 0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1, where=confusion_matrix.sum(axis=1) != 0)
        precision_mean = np.nanmean(precision)  # Moyenne des précisions (gère les cas avec division par zéro)
        recall_mean = np.nanmean(recall)  # Moyenne des rappels
        f1_mean = 2 * (precision_mean * recall_mean) / (precision_mean + recall_mean) if (precision_mean + recall_mean) > 0 else 0

        return accuracy, precision_mean, recall_mean, f1_mean, confusion_matrix
