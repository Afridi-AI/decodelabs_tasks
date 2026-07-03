"""

Dataset facts (matches Slide 8 "Raw Material: The Iris Benchmark"):
    Samples:    150 (balanced, 50 per class)
    Classes:    3   (Setosa, Versicolor, Virginica)
    Dimensions: 4   (Sepal Length, Sepal Width, Petal Length, Petal Width)
==========================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt

RANDOM_STATE = 42  


def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("=" * 70)
    print("STEP 1: LOAD & UNDERSTAND THE DATASET")
    print("=" * 70)
    print(f"Shape of dataset            : {df.shape}")
    print(f"Classes                     : {list(iris.target_names)}")
    print(f"Samples per class           :\n{df['species'].value_counts()}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nStatistical summary:")
    print(df.describe())
    print()
    return iris, df


def prepare_data(iris):
    X = iris.data
    y = iris.target

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 70)
    print("STEP 2/3: TRAIN-TEST SPLIT + FEATURE SCALING")
    print("=" * 70)
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print("StandardScaler applied -> mean ~ 0, variance ~ 1\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def tune_k(X_train, y_train, k_range=range(1, 21)):
   
    error_rates = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        error_rates.append(1 - scores.mean())

    best_k = list(k_range)[int(np.argmin(error_rates))]

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), error_rates, marker="o", linestyle="--", color="#1f4e79")
    plt.axvline(best_k, color="orange", linestyle=":", label=f"Optimal K = {best_k}")
    plt.title("Tuning the Engine: Choosing K (5-fold CV on training set)")
    plt.xlabel("K Value")
    plt.ylabel("Cross-Validated Error Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("E:\DecodeLabs Intership\Project-2\k_tuning_elbow.png", dpi=150)
    plt.close()

    print("=" * 70)
    print("STEP 4: TUNING K (ELBOW METHOD, cross-validated)")
    print("=" * 70)
    print(f"Best K found: {best_k} (lowest CV error rate = {min(error_rates):.4f})")
    print("Elbow plot saved -> k_tuning_elbow.png\n")
    return best_k


# --------------------------------------------------------------------
# STEP 5 (PROCESS): Apply the KNN classification algorithm
# --------------------------------------------------------------------
def train_model(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE
    model.fit(X_train, y_train)                   # FIT (memorize the map)
    print("=" * 70)
    print("STEP 5: TRAIN THE MODEL (scikit-learn KNN)")
    print("=" * 70)
    print(f"model = KNeighborsClassifier(n_neighbors={k})")
    print("model.fit(X_train, y_train)  -> done\n")
    return model


# --------------------------------------------------------------------
# STEP 6 (OUTPUT): Validate with Confusion Matrix + F1 Score
# --------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, target_names):
    predictions = model.predict(X_test)            # PREDICT (apply logic)

    acc = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    print("=" * 70)
    print("STEP 6: OUTPUT VALIDATION")
    print("=" * 70)
    print(f"Accuracy               : {acc:.4f}")
    print(f"F1 Score (macro avg)   : {f1_macro:.4f}\n")

    print("Confusion Matrix (rows = actual, cols = predicted):")
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)
    print("\nFull classification report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    # Save confusion matrix as an image
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Iris KNN Classifier")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("E:\DecodeLabs Intership\Project-2\confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix plot saved -> confusion_matrix.png")

    return acc, f1_macro, cm


# --------------------------------------------------------------------
# MAIN PIPELINE (mirrors Slide 17: "The Full Architecture")
# --------------------------------------------------------------------
def main():
    iris, df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(iris)
    best_k = tune_k(X_train, y_train)
    model = train_model(X_train, y_train, best_k)
    evaluate_model(model, X_test, y_test, iris.target_names)

    print("=" * 70)
    print("COMPLETE - Project 2: Data Classification Using AI")
    print("=" * 70)


if __name__ == "__main__":
    main()
