# Pump Condition Classification
# Algorithms: Decision Tree, Random Forest, SVM, Naive Bayes, AdaBoost

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

def read_dataset(csv_path):
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # Read the file supplied by the user
    data = pd.read_csv(csv_path)

    print("\nDataset loaded successfully.")
    print("Dataset shape:", data.shape)
    print("\nFirst five records:")
    print(data.head())

    return data


# --------------------------------------------------
# 2. Prepare the data
# --------------------------------------------------

def prepare_data(data):
    target = "Pump_Condition"

    if target not in data.columns:
        raise ValueError(f"Missing target column: {target}")

    X = data.drop(columns=[target]).copy()
    y = data[target].copy()

    # Convert text labels into numbers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Keep the label names for evaluation
    class_names = list(label_encoder.classes_)

    # Convert feature columns to numeric values
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace missing values with the median
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )

    return X, y, class_names


# --------------------------------------------------
# 3. Create the five machine learning models
# --------------------------------------------------

def make_models():
    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]),

        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB())
        ]),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    return models


# --------------------------------------------------
# 4. Evaluate each model
# --------------------------------------------------

def evaluate_model(name, model, X_test, y_test, class_names):
    predictions = model.predict(X_test)

    # Find the numeric label representing Fault
    if "Fault" not in class_names:
        raise ValueError("The dataset must contain a 'Fault' class.")

    fault_label = class_names.index("Fault")

    # Calculate the four requested evaluation outputs
    fault_recall = recall_score(
        y_test,
        predictions,
        labels=[fault_label],
        average="macro",
        zero_division=0
    )

    fault_precision = precision_score(
        y_test,
        predictions,
        labels=[fault_label],
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=range(len(class_names))
    )

    print(f"\n{'=' * 45}")
    print(name)
    print(f"{'=' * 45}")

    print(f"Fault Recall:    {fault_recall:.4f}")
    print(f"Fault Precision: {fault_precision:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names
    ))

    # Display the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Condition")
    plt.ylabel("Actual Condition")
    plt.tight_layout()
    plt.savefig(
        f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    plt.show()
    plt.close()

    return {
        "Model": name,
        "Fault Recall": fault_recall,
        "Fault Precision": fault_precision,
        "Macro F1": macro_f1
    }


# --------------------------------------------------
# 5. Train and compare all models
# --------------------------------------------------

def run_classification(csv_path):
    data = read_dataset(csv_path)

    X, y, class_names = prepare_data(data)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    models = make_models()

    # Five-fold cross-validation
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Evaluate Macro F1 using cross-validation
        cv_scores = cross_val_score(
            clone(model),
            X_train,
            y_train,
            cv=cross_validation,
            scoring="f1_macro"
        )

        print(
            f"Cross-validation Macro F1: "
            f"{cv_scores.mean():.4f} "
            f"(+/- {cv_scores.std():.4f})"
        )

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate using the test dataset
        result = evaluate_model(
            name,
            model,
            X_test,
            y_test,
            class_names
        )

        results.append(result)

    # Compare the five models
    results_table = pd.DataFrame(results)

    print("\n\nFINAL MODEL COMPARISON")
    print(results_table.to_string(index=False))

    # Save the comparison table
    results_table.to_csv(
        "pump_model_comparison.csv",
        index=False
    )

    # Plot Macro F1 scores
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=results_table,
        x="Model",
        y="Macro F1"
    )

    plt.title("Model Comparison - Macro F1")
    plt.xlabel("Machine Learning Algorithm")
    plt.ylabel("Macro F1 Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("model_macro_f1_comparison.png")
    plt.show()
    plt.close()

    print("\nAll models have been evaluated.")
    print("Results saved in pump_model_comparison.csv")


# --------------------------------------------------
# 6. Main program
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pump Condition Classification"
    )

    parser.add_argument(
        "--data",
        default="pump_condition_dataset.csv",
        help="Path to the pump dataset CSV file"
    )

    args = parser.parse_args()

    run_classification(args.data)