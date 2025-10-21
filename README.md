# banking77

This project compares full fine-tuning and LoRA (Low-Rank Adaptation) of `bert-base-uncased` on the [Banking77 dataset](https://huggingface.co/datasets/banking77).  
It explores how LoRA rank, learning rate, and scheduler choice affect performance, convergence, and training efficiency on **Apple M4 Pro (MPS GPU)** hardware.

---

## Objective

To evaluate whether LoRA can match the performance of full fine-tuning while training significantly fewer parameters — and to understand how hyperparameters (rank, LR, scheduler) influence this trade-off.

---

## Environment Setup

Clone or download the repo, then install dependencies:

```bash
pip install -r requirements.txt
```
Required libraries:
```
torch
transformers
datasets
evaluate
peft
scikit-learn
matplotlib
```

## Training

Full fine-tuning:
```
python src/train.py --model_name bert-base-uncased
```

LoRA runs (examples):
```
# best configuration
python src/train.py --model_name bert-base-uncased-lora --lr_scheduler cosine

# higher rank
python src/train.py --model_name bert-base-uncased-lora --r 18

# smaller learning rate (for comparison)
python src/train.py --model_name bert-base-uncased-lora --r 8 --lr 1e-4
```

## Evaluation

After training, evaluate saved models:
```
python src/evaluate.py --model_name bert-base-uncased
python src/evaluate.py --model_name bert-base-uncased-loracosine
```

All model checkpoints and metrics are written to artifacts/ (excluded from git).
Aggregated metrics can be found in results/results_summary.csv.


| Run                      | Mode | Rank |   LR   | Scheduler | Trainable Params | Total Params | Share % | Train Time (s) | Val F1 (macro) | Test F1 (macro) |
| :----------------------- | :--: | ---: | :----: | :-------: | ---------------: | -----------: | ------: | -------------: | -------------: | --------------: |
| **BERT-base (full FT)**  | full |    — |    —   |     —     |      109,541,453 |  109,541,453 |  100.00 |         851.45 |     **0.9283** |      **0.9351** |
| **LoRA (r=32)**          | lora |   32 |    —   |     —     |        5,500,493 |  114,958,234 |    4.78 |         951.50 |         0.9204 |          0.9215 |
| **LoRA — cosine**        | lora |   32 |    —   |   cosine  |        5,500,493 |  114,958,234 |    4.78 |        1959.38 |         0.9176 |          0.9225 |
| **LoRA — constant**      | lora |   32 |    —   |  constant |        5,500,493 |  114,958,234 |    4.78 |         960.38 |         0.8613 |          0.8617 |
| **LoRA r=18**            | lora |   18 |    —   |     —     |        3,156,557 |  112,614,298 |    2.80 |         958.34 |         0.9146 |      **0.9236** |
| **LoRA r=9**             | lora |    9 |    —   |     —     |        1,649,741 |  111,107,482 |    1.48 |         988.34 |         0.9103 |          0.9074 |
| **LoRA r=4**             | lora |    4 |    —   |     —     |          812,621 |  110,270,362 |    0.74 |         954.81 |         0.8965 |          0.9095 |
| **LoRA r=32, lr=1.5e-3** | lora |   32 | 1.5e-3 |     —     |        5,500,493 |  114,958,234 |    4.78 |         969.59 |         0.9104 |          0.9192 |
| **LoRA r=32, lr=2e-3**   | lora |   32 |  2e-3  |     —     |        5,500,493 |  114,958,234 |    4.78 |         965.80 |         0.8877 |          0.8794 |
| **LoRA r=32, lr=1e-4**   | lora |   32 |  1e-4  |     —     |        5,500,493 |  114,958,234 |    4.78 |        1282.77 |     **0.7624** |      **0.7613** |


## Insights

Learning rate sensitivity:
Smaller rates (1e-4) drastically reduced performance, confirming LoRA’s need for higher learning rates — typically 10× larger than full fine-tuning — to compensate for its restricted update space.

Rank:
A rank of 18 was enough to reach full fine-tuning quality. Lower ranks degraded F1; higher ranks showed diminishing returns.

Schedulers:
The cosine scheduler gave smooth convergence but slightly increased wall-clock time.

Training time (M4 Pro GPU):
Despite updating only ~5 % of weights, LoRA was not consistently faster.
On Apple’s M4 Pro GPU (Metal backend), adapter layers execute as small sequential kernels.
As discussed in “LoRA Is Slower Than You Think” (Ko, 2025), these tiny adapter operations break GPU parallelism — so LoRA’s FLOP savings don’t always translate into faster wall-clock training.
The benefit is primarily parameter efficiency, not raw speed, especially on non-CUDA GPUs.

## Notes

Model checkpoints are saved under artifacts/ and excluded from git.

Light results (results/*.csv, notebooks/*.ipynb) are committed for reproducibility.

