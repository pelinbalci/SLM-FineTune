#!/usr/bin/env python
"""
Distributed Fine-tuning Script with Accelerate + PEFT

Usage:
    accelerate launch --config_file configs/deepspeed_zero2_config.yaml train_distributed.py
"""

import os
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import time

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./outputs/distributed_finetuned"
MAX_LENGTH = 512
BATCH_SIZE_PER_GPU = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
SEED = 42

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Print distributed info
    if accelerator.is_main_process:
        print("=" * 60)
        print("DISTRIBUTED TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Model: {MODEL_NAME}")
        print(f"Batch size per GPU: {BATCH_SIZE_PER_GPU}")
        print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        effective_batch = BATCH_SIZE_PER_GPU * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS
        print(f"Effective batch size: {effective_batch}")
        print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset
    if accelerator.is_main_process:
        print("\nLoading dataset...")
    
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")
    
    def format_instruction(example):
        if example.get("input", "").strip():
            text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        else:
            text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
        return {"text": text}
    
    formatted_dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Split dataset
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    if accelerator.is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Eval samples: {len(eval_dataset)}")
    
    # Load model
    if accelerator.is_main_process:
        print("\nLoading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Apply LoRA
    if accelerator.is_main_process:
        print("\nApplying LoRA...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    
    if accelerator.is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,
        per_device_eval_batch_size=BATCH_SIZE_PER_GPU,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train!
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    training_time = end_time - start_time
    
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Save model
        print("\nSaving model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
