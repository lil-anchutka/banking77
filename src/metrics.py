
def compute_metrics(evals):
    from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, recall_score, precision_score 
    import numpy as np

    pred_logits, true = evals

    pred = np.argmax(pred_logits, axis=1)

    return {
        'accuracy': accuracy_score(true, pred),
        'balanced_accuracy': balanced_accuracy_score(true, pred),
        'precision_macro': precision_score(true, pred, average='macro', zero_division=0),
        'recall_macro': recall_score(true, pred, average='macro', zero_division=0),
        'f1_macro': f1_score(true, pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(true, pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(true, pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(true, pred, average='weighted', zero_division=0)
    }
