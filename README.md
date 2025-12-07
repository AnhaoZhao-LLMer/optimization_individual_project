# Optimization Individual Project (LoRA Distillation)

This repository contains my **individual project for the Optimization course**, focusing on the **comparison of different optimizers** in the LoRA-based distillation training of Large Reasoning Models (LRMs). The project studies the impact of different optimizers such as **Adam, AdamW, Adafactor, and SGD** on training stability, convergence speed, and reasoning performance.

---

## 🔹 Project Overview

- **Task**: Knowledge distillation for reasoning models  
- **Method**: LoRA-based fine-tuning  
- **Student Model**: 1.5B  
- **Teacher Model**: 7B  
- **Dataset**: GSM8K  
- **Optimizers Compared**:
  - Adam  
  - AdamW  
  - Adafactor  
  - SGD with Momentum  

---

## 🔹 How to Run the Training

Training is controlled by a single script:

```bash
train.sh
```

You only need to modify:

- Model path  
- Dataset path  
- Optimizer type  

Then run:

```bash
bash train.sh
```

---

## 🔹 Key Configuration in train.sh

You need to modify the paths to match your local model and dataset locations, and choose the optimizer you want to test:

```bash
OPTIMIZER_TYPE="Adafactor"

python train.py \
  --model_name_or_path /path/to/student-model-1.5B/ \
  --data_path /path/to/gsm8k/train.jsonl \
  --output_dir SFT-Train/${OPTIMIZER_TYPE} \
  --strategy lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --optimizer_type ${OPTIMIZER_TYPE}
```

---

## 🔹 How to Test Different Optimizers

Simply change:

```bash
OPTIMIZER_TYPE="Adafactor"
```

to one of the following:

```bash
OPTIMIZER_TYPE="Adam"
OPTIMIZER_TYPE="AdamW"
OPTIMIZER_TYPE="Adafactor"
OPTIMIZER_TYPE="SGD"
```

Then re-run:

```bash
bash train.sh
```

Each optimizer will be saved to a separate output directory automatically.

---

## 🔹 Environment Requirements

- Python 3.8+  
- PyTorch  
- HuggingFace Transformers  
- PEFT (LoRA)  
- CUDA-supported GPU (recommended)  

---

## 🔹 Output

All trained models and logs will be saved to:

```bash
SFT-Train/${OPTIMIZER_TYPE}
```

---

## 🔹 Author

Anhao Zhao  
Optimization Course Individual Project  
The Hong Kong Polytechnic University
