
import os
import json
import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import PeftModel
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from tokenization import get_tokenizer
from data import get_banking_data
from metrics import compute_metrics


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("--param_changed", type=str, default='')
    args = parser.parse_args()
    model_name = args.model_name
    param_changed = args.param_changed

    model_path   = f"artifacts/{model_name+param_changed}/best_model"

    eval_dir = f"artifacts/eval/{model_name+param_changed}"
    os.makedirs(eval_dir, exist_ok=True)

    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora:
        model_name = model_name.replace('-lora','')
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=77
        )
        model = PeftModel.from_pretrained(model, model_path)

    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)         

    tokenizer, apply_tokenizer = get_tokenizer(model_name)
    dataset = get_banking_data(0.1, apply_tokenizer)

    training_args = TrainingArguments(output_dir="artifacts/eval_temp", 
    per_device_eval_batch_size=16,
    )

    trainer = Trainer(
    model=model,
    args=training_args,                
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )

    label_names = dataset['test'].features['label'].names

    preds_output = trainer.predict(dataset['test'])
    logits  = preds_output.predictions
    labels  = preds_output.label_ids
    metrics = preds_output.metrics
    pred_labels = logits.argmax(axis=1)


    # --------- usual metrics ---------
    save_dir_m_eval = os.path.join(eval_dir, "metrics_test.json")
    json.dump(metrics, open(save_dir_m_eval, "w"), indent=2)

    # --------- per-class metrics ---------
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, pred_labels, labels=range(len(label_names))
    )
    df = pd.DataFrame({
        "label": label_names,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support
    })
    df.to_csv(os.path.join(eval_dir, "per_class_f1.csv"), index=False)

    # --------- confusion matrix ---------

    cm = confusion_matrix(labels, pred_labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(16, 12)) 
    sns.heatmap(cm_norm, cmap="Blues", square=True,
                xticklabels=False, yticklabels=False,
                cbar=True, linewidths=0.5)

    plt.title(f"Confusion Matrix — {model_name}", fontsize=18)
    plt.tight_layout()

    plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    ### ================== PRINTING THE RESULTS ======================
    print("\n" + "=" * 65)
    print(f"Evaluation Summary — {model_name}")
    print("=" * 65)
    print(f"Model path:         {model_path}")
    print(f"LoRA:               {'Yes' if is_lora else 'No'}")
    print(f"Samples evaluated:  {len(labels)}\n")

    for k, v in metrics.items():
        try:
            print(f"{k:<25}: {float(v):.4f}")
        except Exception:
            print(f"{k:<25}: {v}")

    print("\n" + "-" * 65)
    print("Artifacts saved to:")
    print(f"  • metrics JSON:     {save_dir_m_eval}")
    print(f"  • per-class F1 CSV: {os.path.join(eval_dir, 'per_class_f1.csv')}")
    print(f"  • confusion matrix: {os.path.join(eval_dir, 'confusion_matrix.png')}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
