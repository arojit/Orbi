#!/usr/bin/env python3
"""
Train a LoRA SFT on an ungated model using TRL 0.23.0 (no `tokenizer=`; use `processing_class=`).
- Works with TinyLlama / Qwen2.5 / Phi-3 Mini out of the box
- Uses HF_TOKEN from environment if present
- Handles local JSONL or HF dataset repo
- Formats chat-style `messages` via tokenizer.apply_chat_template when available

Usage examples:
  MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" DS_NAME="orbi_all_dataset_v1.jsonl" python train_orbi_lora_trl023.py
  HF_TOKEN=hf_xxx MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0" python train_orbi_lora_trl023.py
"""

import os
import sys
import math
import torch
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# -----------------------------
# Config (env-overridable)
# -----------------------------
# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DS_NAME = os.environ.get("DS_NAME", "orbi_all_dataset_v1_5k_stories.jsonl")  # local jsonl or HF hub path
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "orbi-1b-lora")
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional; use if gated or private

# Conservative defaults for CPU/low-memory boxes
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
DEVICE = "cuda" if use_cuda else ("mps" if use_mps else "cpu")

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

supports_bf16 = (DEVICE == "cuda") or (DEVICE == "mps" and torch.__version__ >= "2.3")
dtype = torch.bfloat16 if supports_bf16 else (torch.float16 if DEVICE == "cuda" else torch.float32)

MAX_LEN = int(os.environ.get("MAX_LEN", "1024" if DEVICE == "cpu" else "2048"))
EPOCHS = float(os.environ.get("EPOCHS", "1.0" if DEVICE == "cpu" else "2.0"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACC = int(os.environ.get("GRAD_ACC", "16"))
LR = float(os.environ.get("LR", "1e-4"))
WARMUP = float(os.environ.get("WARMUP", "0.03"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "500"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "500"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "25"))

# -----------------------------
# Utilities
# -----------------------------

def guess_lora_targets(model) -> List[str]:
    """Heuristic LoRA target selection across popular architectures.
    Returns list of module name suffixes to target.
    """
    name = model.config.__class__.__name__.lower()
    arch = getattr(model.config, "architectures", None)
    arch = [a.lower() for a in arch] if arch else []
    text = f"{name} {' '.join(arch)}"

    # LLaMA / Gemma style
    if any(k in text for k in ["llama", "gemma"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    # Qwen2 / Qwen2.5
    if "qwen" in text:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    # Phi family (names vary a bit; gather linears containing these hints)
    if "phi" in text:
        linear_names = []
        for n, m in model.named_modules():
            if m.__class__.__name__ == "Linear" and any(s in n for s in ["q", "k", "v", "o", "fc", "mlp", "W", "pack", "proj"]):
                linear_names.append(n.split(".")[-1])
        dedup = sorted(set([n for n in linear_names if len(n) < 40]))
        return dedup or ["W_pack", "out_proj", "fc1", "fc2"]
    # GPT-NeoX / GPT-J / GPT-2 like
    if any(k in text for k in ["gptneox", "gptj", "gpt2"]):
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "c_attn", "c_proj"]

    # Fallback: all linear layer short names
    linear_names = []
    for n, m in model.named_modules():
        if m.__class__.__name__ == "Linear":
            linear_names.append(n.split(".")[-1])
    return sorted(set([n for n in linear_names if len(n) < 40]))


def build_text(tokenizer, ex: dict) -> str:
    """Format a single example to plain text for SFT.
    - If `messages` present, use chat template.
    - Else use `text`, or join common instruction fields.
    """
    if "messages" in ex and ex["messages"]:
        return tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
    if "text" in ex and ex["text"]:
        return ex["text"]
    parts = []
    for k in ["instruction", "input", "output", "answer", "response"]:
        if k in ex and ex[k]:
            parts.append(f"{k.upper()}:\n{ex[k]}")
    return "\n\n".join(parts)


# -----------------------------
# Load tokenizer & model
# -----------------------------
print(f"Loading tokenizer: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"Loading model: {MODEL_ID} (dtype={dtype}, device={DEVICE})")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map=("auto" if DEVICE != "cpu" else None),
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
)
if DEVICE == "cpu":
    model.to("cpu")

# LoRA config
print("Choosing LoRA target modules…")
loras = guess_lora_targets(model)
print(f"LoRA targets: {loras[:12]}{' …' if len(loras)>12 else ''}")

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=loras,
    task_type="CAUSAL_LM",
)

# -----------------------------
# Load dataset (local jsonl or HF repo)
# -----------------------------
print(f"Loading dataset: {DS_NAME}")
if os.path.exists(DS_NAME):
    ds = load_dataset("json", data_files={"train": DS_NAME})
    # make a tiny eval split from train if nothing else is present
    eval_count = min(64, len(ds["train"]))
    ds["test"] = ds["train"].select(range(eval_count))
else:
    ds = load_dataset(DS_NAME)

train = ds["train"].map(lambda e: {"text": build_text(tok, e)}, remove_columns=[c for c in ds["train"].column_names if c != "text"]) 

evald = ds.get("validation") or ds.get("test")
if evald is None:
    evald = ds["train"].select(range(min(64, len(ds["train"]))))

evald = evald.map(lambda e: {"text": build_text(tok, e)}, remove_columns=[c for c in evald.column_names if c != "text"]) 

# -----------------------------
# TRL 0.23.0 Trainer (use processing_class, not tokenizer)
# -----------------------------
print("Configuring trainer…")
cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=MAX_LEN,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    warmup_ratio=WARMUP,
    lr_scheduler_type="cosine",
    packing=True,
    gradient_checkpointing=(DEVICE != "cpu"),
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    peft_config=lora,
    train_dataset=train,
    eval_dataset=evald,
    args=cfg,
)

# -----------------------------
# Train & Save
# -----------------------------
print("Starting training…")
train_result = trainer.train()
print("Training done.")

print(f"Saving to {OUTPUT_DIR}…")
trainer.model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print("All done.")
