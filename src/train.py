
import os
import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
import numpy as np
import json

from tokenization import get_tokenizer
from data import get_banking_data
from metrics import compute_metrics
import warnings


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train")
    parser.add_argument("--lr_rate", type=float)
    parser.add_argument("--lr_scheduler_type", type=str)
    parser.add_argument("--r", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16"])
    parser.add_argument("--param_changed", type=str, default='')

    args = parser.parse_args()

    param_changed = args.param_changed
    model_name = args.model_name
    model_config_path = f"artifacts/configs/{model_name}.json"

    with open(model_config_path, "r") as f:
        config = json.load(f)

    if args.lr_scheduler_type is not None:
        config["training_args"]["lr_scheduler_type"] = args.lr_scheduler_type
    if args.r is not None:
        config["lora"]["r"] = args.r
    if args.lr_rate is not None:
        config["training_args"]["learning_rate"] = args.lr_rate
    if args.precision is not None:
        config["training_args"]["bf16"] = (args.precision == "bf16")
        config["training_args"]["fp16"] = False

    tokenizer, apply_tokenizer = get_tokenizer(config['model_name'])
    dataset = get_banking_data(0.1, apply_tokenizer)

    id2labels = dict([(k, v) for (k, v) in enumerate(dataset["train"].features["label"].names)])

    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=77, id2label = id2labels)

    if config["lora"]["enabled"]:
        lora_args = {k: v for k, v in config["lora"].items() if k != "enabled"}
        lora_cfg = LoraConfig(**lora_args)
        model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(**config["training_args"])

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    train_output = trainer.train()

    save_dir = f"artifacts/{model_name+param_changed}/best_model"
    os.makedirs(save_dir, exist_ok=True)

    trainer.save_model(save_dir)
    trainer.model.config.save_pretrained(save_dir)

    save_dir_m = f"artifacts/{model_name+param_changed}/train_metrics.json"

    json.dump(train_output.metrics, open(save_dir_m, "w"), indent=2)

    val_metrics = trainer.evaluate(eval_dataset=dataset["valid"])

    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())

    num_train_samples = len(trainer.train_dataset)
    num_eval_samples = len(trainer.eval_dataset)


    ### ================== PRINTING THE RESULTS ======================
    print("\n" + "=" * 65)
    print(f"Training Summary — {model_name}")
    print("=" * 65)
    print(f"Model path:         artifacts/{model_name}/best_model")
    print(f"LoRA:               {'Yes' if config['lora']['enabled'] else 'No'}")
    print(f"Training time:      {train_output.metrics.get('train_runtime', 0):.2f} sec")
    print(f"Trainable params: {trainable_params:,} / {total_params:,} "
      f"({trainable_params/total_params*100:.2f}%)")
    print(f"Train samples:      {num_train_samples:,}")
    print(f"Validation samples: {num_eval_samples:,}")
    print(f"Best checkpoint:    {trainer.state.best_model_checkpoint}")
    if trainer.state.best_metric is not None:
        print(f"Best metric ({config['training_args']['metric_for_best_model']}): "
              f"{trainer.state.best_metric:.4f}")
    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        try:
            print(f"{k:<25}: {float(v):.4f}")
        except Exception:
            print(f"{k:<25}: {v}")

    print("\n" + "-" * 65)
    print("Artifacts saved to:")
    print(f"  • model:          artifacts/{model_name+param_changed}/best_model")
    print(f"  • train metrics:  {save_dir_m}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
