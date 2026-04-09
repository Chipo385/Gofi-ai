"""
finetune_gemma.py
=================
Gofi AI — Q-LoRA Fine-Tuning for Gemma on Zambian Financial Data
Uses Unsloth for ultra-fast LoRA fine-tuning (2x speed vs standard).
Falls back to PEFT + BitsAndBytes if Unsloth is unavailable.

Training Data Format (JSONL):
    {"instruction": "...", "input": "...", "output": "..."}

Example:
    {"instruction": "Analyze the impact of copper price drops on ZCCM-IH stock",
     "input": "Copper fell 8% this week to $8,200/tonne.",
     "output": "ZCCM-IH (ZCCM.LUSE) is highly correlated with global copper prices..."}

Usage:
    # Install dependencies first
    pip install unsloth peft trl bitsandbytes datasets transformers

    # Train
    python finetune_gemma.py --train_file data/training_data.jsonl
    
    # Train with custom settings
    python finetune_gemma.py \\
        --train_file data/training_data.jsonl \\
        --model_id google/gemma-2b-it \\
        --epochs 3 \\
        --lr 2e-4 \\
        --rank 16 \\
        --output_dir ./gofi-gemma-lora
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# Try Unsloth first (faster); fall back to standard PEFT
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("[Train] Using Unsloth for accelerated training.")
except ImportError:
    UNSLOTH_AVAILABLE = False
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    print("[Train] Unsloth not found. Using standard PEFT.")

from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Config Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID  = "google/gemma-2b-it"
DEFAULT_OUTPUT    = "./gofi-gemma-lora"
DEFAULT_RANK      = 16         # LoRA rank — higher = more params, better quality
DEFAULT_ALPHA     = 32         # LoRA scaling factor (usually 2x rank)
DEFAULT_DROPOUT   = 0.05
DEFAULT_LR        = 2e-4
DEFAULT_EPOCHS    = 3
DEFAULT_BATCH     = 2          # Increase if you have more VRAM
DEFAULT_GRAD_ACC  = 4          # Effective batch = BATCH * GRAD_ACC
DEFAULT_MAX_SEQ   = 2048       # Max token length per training example

# Target modules for LoRA (Gemma architecture)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# 1. Prompt Formatting
# ---------------------------------------------------------------------------

ALPACA_PROMPT = """Below is an instruction from a Zambian financial analyst. Write a response that thoroughly addresses the request.

### Instruction:
{instruction}

### Context:
{input}

### Response:
{output}"""


def format_prompt(example: dict) -> dict:
    """Formats a JSONL training example into the Alpaca instruction template."""
    text = ALPACA_PROMPT.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", "No additional context provided."),
        output=example.get("output", ""),
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# 2. Data Loading
# ---------------------------------------------------------------------------

def load_training_data(train_file: str) -> Dataset:
    """
    Loads JSONL training data and formats it for SFT.
    Expects each line: {"instruction": "...", "input": "...", "output": "..."}
    """
    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    records = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    dataset = Dataset.from_list(records)
    dataset = dataset.map(format_prompt)
    print(f"[Train] Loaded {len(dataset)} training examples from {train_file}")
    return dataset


# ---------------------------------------------------------------------------
# 3a. Model Loading — Unsloth Path (Recommended)
# ---------------------------------------------------------------------------

def load_model_unsloth(model_id: str, rank: int, max_seq: int):
    """Fast model + LoRA loading using Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq,
        load_in_4bit=True,          # Q4 quantization
        dtype=None,                  # Auto-select (bfloat16 on Ampere+)
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=DEFAULT_ALPHA,
        lora_dropout=DEFAULT_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Saves ~30% VRAM
        random_state=42,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# 3b. Model Loading — Standard PEFT Path (Fallback)
# ---------------------------------------------------------------------------

def load_model_peft(model_id: str, rank: int):
    """Standard LoRA loading (quantized on CUDA, float16 on MPS)."""
    # Detect Apple Silicon
    is_mac = torch.backends.mps.is_available()

    if is_mac:
        print("[Train] Apple Silicon (MPS) detected. Bypassing 4-bit quantization (BitsAndBytes).")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        print("[Train] CUDA detected. Using 4-bit BitsAndBytes quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NF4 = better quality at 4-bit
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,       # Double quantization to save memory
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=DEFAULT_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=DEFAULT_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------

def train(
    train_file: str,
    model_id:   str = DEFAULT_MODEL_ID,
    output_dir: str = DEFAULT_OUTPUT,
    epochs:     int = DEFAULT_EPOCHS,
    lr:         float = DEFAULT_LR,
    rank:       int = DEFAULT_RANK,
    batch_size: int = DEFAULT_BATCH,
    grad_acc:   int = DEFAULT_GRAD_ACC,
    max_seq:    int = DEFAULT_MAX_SEQ,
):
    """
    Main fine-tuning loop using HuggingFace TRL's SFTTrainer.
    Works on consumer GPUs (e.g., RTX 3080) with 4-bit quantization.
    Estimated time: ~2-4 hours for 500 examples on a single A100.
    """
    # --- Load data ---
    dataset = load_training_data(train_file)

    # --- Load model ---
    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_unsloth(model_id, rank, max_seq)
    else:
        model, tokenizer = load_model_peft(model_id, rank)

    # Auto-detect best memory configuration depending on device
    is_mac = torch.backends.mps.is_available()
    optim_algo = "adamw_torch" if is_mac else "adamw_8bit"

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        lr_scheduler_type="cosine",         # Cosine decay
        warmup_ratio=0.05,                  # 5% warmup
        weight_decay=0.01,
        fp16=False if is_mac else not torch.cuda.is_bf16_supported(),
        bf16=False if is_mac else torch.cuda.is_bf16_supported(),
        optim=optim_algo,                   # 8-bit Adam for CUDA, standard for MPS
        logging_steps=10,
        eval_strategy="no", # Note: evaluation_strategy is deprecated
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",                   # Set to "wandb" if you want tracking
        seed=42,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq,
        args=training_args,
        packing=True,                       # Pack multiple short examples per batch
    )

    # --- Train ---
    print(f"\n[Train] Starting fine-tuning: {epochs} epoch(s), LR={lr}, LoRA rank={rank}")
    print(f"[Train] Output will be saved to: {output_dir}\n")
    trainer.train()

    # --- Save ---
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[Train] ✓ Fine-tuning complete! Model saved to: {output_dir}")
    print("[Train] Merge LoRA weights before serving:")
    print("        model.merge_and_unload() then model.save_pretrained('./gofi-merged')")


# ---------------------------------------------------------------------------
# 5. Merge & Export (run after training)
# ---------------------------------------------------------------------------

def merge_lora_weights(lora_dir: str, output_dir: str):
    """
    Merges LoRA adapter weights into the base model for efficient inference.
    Run this after training to create a single deployable model.
    """
    from peft import PeftModel

    print(f"[Merge] Loading base model + LoRA from {lora_dir}...")
    base_id = DEFAULT_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.merge_and_unload()

    print(f"[Merge] Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Merge] ✓ Merged model saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gofi AI — Fine-Tune Gemma for LuSE")
    parser.add_argument("--train_file",  type=str,   default="data/training_data.jsonl")
    parser.add_argument("--model_id",    type=str,   default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir",  type=str,   default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--rank",        type=int,   default=DEFAULT_RANK)
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--grad_acc",    type=int,   default=DEFAULT_GRAD_ACC)
    parser.add_argument("--max_seq",     type=int,   default=DEFAULT_MAX_SEQ)
    parser.add_argument("--merge",       action="store_true",
                        help="Merge LoRA weights after training")
    parser.add_argument("--merge_output", type=str, default="./gofi-merged")

    args = parser.parse_args()

    train(
        train_file=args.train_file,
        model_id=args.model_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        batch_size=args.batch_size,
        grad_acc=args.grad_acc,
        max_seq=args.max_seq,
    )

    if args.merge:
        merge_lora_weights(args.output_dir, args.merge_output)
