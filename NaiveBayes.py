import numpy as np

class NaiveBayes:
    def __init__(self):
        """
        Initializer pour le classifieur Naive Bayes.
        """
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def train(self, train, train_labels):
        """
        Entraîne le classifieur sur l'ensemble d'entraînement.
        Args:
            train: Matrice des exemples d'entraînement (numpy array).
            train_labels: Labels associés aux exemples (numpy array).
        """
        self.classes = np.unique(train_labels)
        self.mean = np.zeros((len(self.classes), train.shape[1]))
        self.var = np.zeros((len(self.classes), train.shape[1]))
        self.priors = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):
            X_c = train[train_labels == c]
            self.mean[idx, :] = np.mean(X_c, axis=0)
            self.var[idx, :] = np.var(X_c, axis=0) + 1e-9  # Pour éviter une variance nulle
            self.priors[idx] = X_c.shape[0] / train.shape[0]

    def predict(self, x):
        """
        Prédire la classe d'un exemple donné.
        Args:
            x: Exemple à prédire (numpy array).
        Retours:
            Classe prédite (int).
        """
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[idx]))
            likelihood -= 0.5 * np.sum(((x - self.mean[idx]) ** 2) / self.var[idx])
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

    def evaluate(self, X, y):
        """
        Évalue le classifieur sur un ensemble donné.
        Args:
            X: Matrice des exemples de test (numpy array).
            y: Labels des exemples de test (numpy array).
        Retours:
            accuracy: Accuracy du modèle (float).
            precision: Précision moyenne (float).
            recall: Rappel moyen (float).
            f1: F1-score moyen (float).
            confusion_matrix: Matrice de confusion (numpy array).
        """
        predictions = np.array([self.predict(x) for x in X])
        accuracy = np.mean(predictions == y)

        # Calculer la matrice de confusion
        unique_labels = np.unique(np.concatenate((y, predictions)))
        confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        for true_label, pred_label in zip(y, predictions):
            confusion_matrix[label_to_index[true_label], label_to_index[pred_label]] += 1

        # Calculer la precision, recall, et F1-score
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0, where=confusion_matrix.sum(axis=0) != 0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1, where=confusion_matrix.sum(axis=1) != 0)
        precision_mean = np.nanmean(precision)  # Gérér les cas de précision non défini
        recall_mean = np.nanmean(recall)  # Gérer les cas de Recall non défini
        f1_mean = 2 * (precision_mean * recall_mean) / (precision_mean + recall_mean) if (precision_mean + recall_mean) > 0 else 0

        return accuracy, precision_mean, recall_mean, f1_mean, confusion_matrix
