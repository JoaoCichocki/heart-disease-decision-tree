"""Run the Heart Disease Decision Tree experiment."""

from sklearn.model_selection import train_test_split

from src.data_loader import load_heart_disease_data
from src.decision_tree import DecisionTree
from src.evaluation import evaluate_classifier


def main():
    X, y, feature_names = load_heart_disease_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = DecisionTree(max_depth=4, min_samples_leaf=2)
    model.fit(X_train, y_train)

    print("=" * 55)
    print("HEART DISEASE - DECISION TREE FROM SCRATCH")
    print("=" * 55)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print()
    evaluate_classifier(model, X_test, y_test)

    print("\nLearned decision tree:")
    model.print_tree(feature_names)

    print("\nFeature importance (normalized):")
    ranked = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True
    )
    for name, importance in ranked:
        print(f"{name:12s} {importance:.4f}")


if __name__ == "__main__":
    main()
