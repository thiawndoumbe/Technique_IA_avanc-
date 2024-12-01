import time
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import load_datasets
import Knn
import NaiveBayes


def afficher_resultats(
    nom_modele: str,
    nom_dataset: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    matrice_confusion: np.ndarray,
    temps_entrainement: float,
    temps_test: float,
    k_optimal: int = None
) -> None:
    """
    Affiche les résultats d'évaluation d'un modèle pour un dataset donné.

    Args:
        nom_modele: Nom du modèle évalué.
        nom_dataset: Nom du dataset utilisé.
        accuracy: Exactitude du modèle (float).
        precision: Précision moyenne (float).
        recall: Rappel moyen (float).
        f1: F1-score moyen (float).
        matrice_confusion: Matrice de confusion (numpy array).
        temps_entrainement: Temps d'entraînement en secondes (float).
        temps_test: Temps de test en secondes (float).
        k_optimal: Si applicable, la valeur optimale de k pour KNN (int).
    """
    print(f"--- Évaluation de {nom_modele} sur le dataset {nom_dataset} ---")
    if k_optimal:
        print(f"K optimal : {k_optimal}")
    print(f"Temps d'entraînement : {temps_entrainement:.4f} secondes")
    print(f"Temps de test : {temps_test:.4f} secondes")
    print(f"Exactitude (Accuracy) : {accuracy:.2f}")
    print(f"Précision : {precision:.2f}")
    print(f"Rappel : {recall:.2f}")
    print(f"Score F1 : {f1:.2f}")
    print("Matrice de confusion :")
    print(matrice_confusion)
    print("\n")


def evaluer_modele_personnalise(
    modele: object,
    donnees_entrainement: np.ndarray,
    etiquettes_entrainement: np.ndarray,
    donnees_test: np.ndarray,
    etiquettes_test: np.ndarray
) -> tuple:
    """
    Évalue un modèle personnalisé sur un dataset donné.

    Args:
        modele: Le modèle personnalisé (ex. Knn ou NaiveBayes).
        donnees_entrainement: Matrice des exemples d'entraînement (numpy array).
        etiquettes_entrainement: Labels des exemples d'entraînement (numpy array).
        donnees_test: Matrice des exemples de test (numpy array).
        etiquettes_test: Labels des exemples de test (numpy array).

    Retours:
        accuracy: Accuracy du modèle (float).
        precision: Précision moyenne (float).
        recall: Rappel moyen (float).
        f1: F1-score moyen (float).
        confusion_matrix: Matrice de confusion (numpy array).
        temps_entrainement: Temps d'entraînement en secondes (float).
        temps_test: Temps de test en secondes (float).
    """
    debut_entrainement = time.time()
    modele.train(donnees_entrainement, etiquettes_entrainement)
    fin_entrainement = time.time()

    debut_test = time.time()
    resultats = modele.evaluate(donnees_test, etiquettes_test)
    fin_test = time.time()

    temps_entrainement = fin_entrainement - debut_entrainement
    temps_test = fin_test - debut_test
    return *resultats, temps_entrainement, temps_test


def evaluer_modele_sklearn(
    modele: object,
    donnees_entrainement: np.ndarray,
    etiquettes_entrainement: np.ndarray,
    donnees_test: np.ndarray,
    etiquettes_test: np.ndarray
) -> tuple:
    """
    Évalue un modèle scikit-learn sur un dataset donné.

    Args:
        modele: Le modèle scikit-learn (ex. GaussianNB ou KNeighborsClassifier).
        donnees_entrainement: Matrice des exemples d'entraînement (numpy array).
        etiquettes_entrainement: Labels des exemples d'entraînement (numpy array).
        donnees_test: Matrice des exemples de test (numpy array).
        etiquettes_test: Labels des exemples de test (numpy array).

    Retours:
        accuracy: Accuracy du modèle (float).
        precision: Précision moyenne (float).
        recall: Rappel moyen (float).
        f1: F1-score moyen (float).
        confusion_matrix: Matrice de confusion (numpy array).
        temps_entrainement: Temps d'entraînement en secondes (float).
        temps_test: Temps de test en secondes (float).
    """
    debut_entrainement = time.time()
    modele.fit(donnees_entrainement, etiquettes_entrainement)
    fin_entrainement = time.time()

    debut_test = time.time()
    predictions = modele.predict(donnees_test)
    fin_test = time.time()

    accuracy = np.mean(predictions == etiquettes_test)

    # Générer la matrice de confusion
    etiquettes_uniques = np.unique(np.concatenate((etiquettes_test, predictions)))
    matrice_confusion = np.zeros((len(etiquettes_uniques), len(etiquettes_uniques)), dtype=int)
    etiquettes_to_index = {label: idx for idx, label in enumerate(etiquettes_uniques)}
    for vrai, predit in zip(etiquettes_test, predictions):
        matrice_confusion[etiquettes_to_index[vrai], etiquettes_to_index[predit]] += 1

    # Calculer précision, rappel et F1-score
    precision = np.diag(matrice_confusion) / np.sum(matrice_confusion, axis=0, where=matrice_confusion.sum(axis=0) != 0)
    recall = np.diag(matrice_confusion) / np.sum(matrice_confusion, axis=1, where=matrice_confusion.sum(axis=1) != 0)
    precision_moyenne = np.nanmean(precision)
    rappel_moyen = np.nanmean(recall)
    f1_moyen = 2 * (precision_moyenne * rappel_moyen) / (precision_moyenne + rappel_moyen) if (precision_moyenne + rappel_moyen) > 0 else 0

    temps_entrainement = fin_entrainement - debut_entrainement
    temps_test = fin_test - debut_test
    return accuracy, precision_moyenne, rappel_moyen, f1_moyen, matrice_confusion, temps_entrainement, temps_test


def validation_croisee_knn(
    donnees_entrainement: np.ndarray,
    etiquettes_entrainement: np.ndarray,
    max_k: int = 20,
    n_splits: int = 5
) -> tuple:
    """
    Effectue une validation croisée pour trouver la valeur optimale de k pour KNN.

    Args:
        donnees_entrainement: Matrice des exemples d'entraînement (numpy array).
        etiquettes_entrainement: Labels des exemples d'entraînement (numpy array).
        max_k: Valeur maximale de k à tester (int).
        n_splits: Nombre de plis (folds) pour la validation croisée (int).

    Retours:
        k_optimal: Valeur optimale de k avec la meilleure précision (int).
        scores: Liste des précisions moyennes pour chaque k (list[float]).
    """
    taille_fold = len(donnees_entrainement) // n_splits
    scores = []

    for k in range(1, max_k + 1):
        precisions_fold = []

        for i in range(n_splits):
            # Diviser les données en folds d'entraînement et de validation
            debut, fin = i * taille_fold, (i + 1) * taille_fold
            val_donnees, val_etiquettes = donnees_entrainement[debut:fin], etiquettes_entrainement[debut:fin]
            train_fold_donnees = np.concatenate((donnees_entrainement[:debut], donnees_entrainement[fin:]))
            train_fold_etiquettes = np.concatenate((etiquettes_entrainement[:debut], etiquettes_entrainement[fin:]))

            # Entraîner et valider le modèle KNN
            knn = Knn.Knn(k=k)
            knn.train(train_fold_donnees, train_fold_etiquettes)
            precision = np.mean([knn.predict(x) == y for x, y in zip(val_donnees, val_etiquettes)])
            precisions_fold.append(precision)

        # Moyenne des précisions pour ce k
        scores.append(np.mean(precisions_fold))

    k_optimal = np.argmax(scores) + 1  # k commence à 1
    return k_optimal, scores


# Paramètres
ratio_entrainement = 0.8

# Chargement des datasets
datasets = [
    ("Iris", *load_datasets.load_iris_dataset(ratio_entrainement)),
    ("Wine Quality", *load_datasets.load_wine_dataset(ratio_entrainement)),
    ("Abalone", *load_datasets.load_abalone_dataset(ratio_entrainement)),
]

resultats_recap = []

for nom_dataset, train_data, train_labels, test_data, test_labels in datasets:
    # Validation croisée pour KNN
    k_optimal, _ = validation_croisee_knn(train_data, train_labels)

    # Modèles
    knn_personnalise = Knn.Knn(k=k_optimal)
    sklearn_knn = KNeighborsClassifier(n_neighbors=k_optimal)
    nb_personnalise = NaiveBayes.NaiveBayes()
    nb_sklearn = GaussianNB()

    # Évaluation KNN personnalisé
    knn_results = evaluer_modele_personnalise(knn_personnalise, train_data, train_labels, test_data, test_labels)
    afficher_resultats("KNN Personnalisé", nom_dataset, *knn_results, k_optimal)
    resultats_recap.append(["KNN Personnalisé", nom_dataset, k_optimal, *knn_results[:4], knn_results[-2], knn_results[-1]])

    # Évaluation KNN scikit-learn
    sklearn_knn_results = evaluer_modele_sklearn(sklearn_knn, train_data, train_labels, test_data, test_labels)
    afficher_resultats("KNN (Scikit-learn)", nom_dataset, *sklearn_knn_results, k_optimal)
    resultats_recap.append(["KNN (Scikit-learn)", nom_dataset, k_optimal, *sklearn_knn_results[:4], sklearn_knn_results[-2], sklearn_knn_results[-1]])

    # Évaluation Naive Bayes personnalisé
    nb_results = evaluer_modele_personnalise(nb_personnalise, train_data, train_labels, test_data, test_labels)
    afficher_resultats("Naive Bayes Personnalisé", nom_dataset, *nb_results)
    resultats_recap.append(["Naive Bayes Personnalisé", nom_dataset, "-", *nb_results[:4], nb_results[-2], nb_results[-1]])

    # Évaluation Naive Bayes scikit-learn
    nb_sklearn_results = evaluer_modele_sklearn(nb_sklearn, train_data, train_labels, test_data, test_labels)
    afficher_resultats("Naive Bayes (Scikit-learn)", nom_dataset, *nb_sklearn_results)
    resultats_recap.append(["Naive Bayes (Scikit-learn)", nom_dataset, "-", *nb_sklearn_results[:4], nb_sklearn_results[-2], nb_sklearn_results[-1]])

# Affichage du tableau récapitulatif
recap_df = pd.DataFrame(
    resultats_recap,
    columns=["Modèle", "Dataset", "K Optimal", "Exactitude", "Précision", "Rappel", "F1-score", "Temps Entraînement (s)", "Temps Test (s)"],
)
print("--- Tableau récapitulatif ---")
print(recap_df)
