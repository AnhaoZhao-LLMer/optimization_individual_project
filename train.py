#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最朴素 SFT 训练脚本（整段 completion 上打 loss）
- 不再是“单 token 分类”，而是对 completion 的所有 token 计算 loss
- 不做样本加权 / token 加权
- 验证集只看 loss（ppl），不再算 easy/hard 之类的指标
- 支持 LoRA / 全量微调
- 数据：jsonl，每行至少包含 {idx, question, answer}
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from torch.optim import Adam, SGD
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROMPT_USER_PREFIX = (
    "<｜begin▁of▁sentence｜><｜User｜>{question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
    "<｜Assistant｜><think>\n"
)

# 可选：PEFT（LoRA）
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# ------------------ 数据工具 ------------------


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def train_eval_split(
    records: List[dict],
    eval_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """最普通的随机划分 train / eval"""
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)

    n_eval = max(1, int(len(records) * eval_ratio))
    eval_idxs = set(idxs[:n_eval])
    train = [records[i] for i in range(len(records)) if i not in eval_idxs]
    eval_ = [records[i] for i in range(len(records)) if i in eval_idxs]

    print(f"[Split] Train={len(train)}, Eval={len(eval_)} (ratio={eval_ratio})")
    return train, eval_


# ------------------ Dataset：对 completion 全部 token 打 loss ------------------


class JsonlSFTDataset(Dataset):
    """
    朴素 SFT：
    - 输入：question 作为 prompt，answer 作为 completion
    - labels：
        - prompt 段全部设为 -100（不算 loss）
        - completion 段全部用真实 token id（算 loss）
    - 会在末尾可选加一个 EOS（也算在 completion 里）
    """

    def __init__(self, records: List[dict], tokenizer: AutoTokenizer, max_length: int):
        self.records = records
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        # 你可以根据自己的需要改成别的 prompt 模板
        question = item["question"]
        answer = item["code"][0]

        # ✅ 用你的模板包一层
        prompt_text = PROMPT_USER_PREFIX.format(question=question)
        completion_text = answer
        
        # 分别 tokenize（不加 BOS/EOS）
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        comp_ids = self.tok(completion_text, add_special_tokens=False)["input_ids"]
        
        eos_id = self.tok.eos_token_id

        # 先拼起来：prompt + completion
        input_ids = prompt_ids + comp_ids

        # 如果还能放 EOS，就在末尾加一个，并把它也视为 completion 的一部分
        if eos_id is not None and len(input_ids) < self.max_length:
            input_ids.append(eos_id)
            comp_ids = comp_ids + [eos_id]

        # 截断到 max_length
        input_ids = input_ids[: self.max_length]

        # 计算 prompt 实际长度（可能被截断）
        prompt_len = min(len(prompt_ids), len(input_ids))

        # labels：prompt 部分为 -100，后面（completion 部分）用真实 token id
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # attention_mask：非 pad 部分全 1（pad 在 collator 里做）
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ------------------ 简单 pad collator ------------------


@dataclass
class SFTDataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:

        # 对 input_ids / attention_mask 用 tokenizer.pad
        
        to_pad = [{k: f[k] for k in ["input_ids", "attention_mask"]} for f in features]
        padded = self.tokenizer.pad(
            to_pad,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # labels 手动 pad（值为 -100）
        max_len = padded["input_ids"].size(1)
        labels = []
        for f in features:
            lab = f["labels"]
            pad_len = max_len - lab.size(0)
            if pad_len > 0:
                if self.tokenizer.padding_side == "left":
                    lab = torch.cat(
                        [torch.full((pad_len,), -100, dtype=lab.dtype), lab]
                    )
                else:
                    lab = torch.cat(
                        [lab, torch.full((pad_len,), -100, dtype=lab.dtype)]
                    )
            labels.append(lab)
        padded["labels"] = torch.stack(labels, dim=0)

        return padded


# ------------------ 主流程 ------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/code/models/R1-Distill-Qwen-7B/",
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--strategy", type=str, choices=["lora", "full"], default="lora"
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--eval_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=None)  # 若 None，则按策略给默认
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument(
        "--bf16", type=lambda x: str(x).lower() == "true", default=True
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=60)
    parser.add_argument("--optimizer_type", type=str, choices=["Adam", "AdamW", "Adafactor", "sgd"], default="AdamW")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = args.max_length

    # 数据
    records = load_jsonl(args.data_path)
    train_recs, eval_recs = train_eval_split(
        records, eval_ratio=args.eval_ratio, seed=args.seed
    )
    print(f"Train size: {len(train_recs)}, Eval size: {len(eval_recs)}")

    train_ds = JsonlSFTDataset(train_recs, tok, args.max_length)
    eval_ds = JsonlSFTDataset(eval_recs, tok, args.max_length)
    collator = SFTDataCollator(tokenizer=tok, pad_to_multiple_of=None)

    # 模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )

    # LoRA or Full
    if args.strategy == "lora":
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft 未找到，请先安装：pip install peft")
        lora_cfg = LoraConfig(
            r=64,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        default_lr = 2e-4
    else:
        default_lr = 2e-5

    lr = args.learning_rate if args.learning_rate is not None else default_lr

    # 如果是 Adafactor，必须在这里声明 optim
    if args.optimizer_type == "Adafactor":
        targs = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=lr,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            bf16=args.bf16,
            logging_steps=args.logging_steps,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            dataloader_pin_memory=True,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            optim="adafactor",   # ✅ 关键
        )
    else:
        # 其他优化器（Adam / AdamW / SGD）走默认逻辑
        targs = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=lr,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            bf16=args.bf16,
            logging_steps=args.logging_steps,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            dataloader_pin_memory=True,
            report_to=["tensorboard"],
            remove_unused_columns=False,
        )

    # ========= 2️⃣ 根据 optimizer_type 构造 Trainer =========

    if args.optimizer_type == "AdamW":
        # ✅ Trainer 默认就是 AdamW，什么都不用传
        print(">>> Using AdamW Optimizer (Trainer Default)")
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=tok,
        )

    elif args.optimizer_type == "Adam":
        print(">>> Using Adam Optimizer")
        optimizer = Adam(
            model.parameters(),
            lr=lr
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=tok,
            optimizers=(optimizer, None),   # ✅ 关键
        )

    elif args.optimizer_type == "sgd":
        print(">>> Using SGD + Momentum Optimizer")
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9   # ✅ 你也可以作为超参暴露出去
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=tok,
            optimizers=(optimizer, None),   # ✅ 关键
        )

    elif args.optimizer_type == "Adafactor":
        print(">>> Using Adafactor Optimizer")
        # ✅ optimizer 已经通过 TrainingArguments 里的 optim="adafactor" 设置好了
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=tok,
        )

    else:
        raise ValueError(f"Unknown optimizer type: {args.optimizer_type}")

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)  # 会包含 eval_loss / eval_runtime 等

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
