"""Model evaluation helpers."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classifier(model, X_test, y_test):
    """Print and return common classification metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0,
        target_names=["No disease (0)", "Disease (1)"]
    ))
    return accuracy
