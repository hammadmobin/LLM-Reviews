from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_predictions(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "report": classification_report(labels, preds)
    }
