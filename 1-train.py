import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ======================================================
# 1) CONFIG (GTX 1050 4GB SAFE)
# ======================================================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "trl-lib/Capybara"
OUT_DIR = "./outputs/qwen2_5_0_5b-capybara-lora"

MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 16
MAX_STEPS = 3
LR = 2e-4

# ======================================================
# 2) LOAD DATASET
# ======================================================
dataset = load_dataset(DATASET_ID, split="train")

# ======================================================
# 3) TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# 4) BASE MODEL
# ======================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
)

# ======================================================
# 5) LoRA
# ======================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================================================
# 6) TRAINING ARGUMENTS
# ======================================================
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    learning_rate=LR,
    fp16=True,
    logging_steps=1,
    save_steps=1,
    save_total_limit=2,
    report_to="none",
    optim="adamw_torch",
)

# ======================================================
# 7) TRAINER (CORRECT API)
# ======================================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ======================================================
# 8) TRAIN
# ======================================================
trainer.train()

# ======================================================
# 9) SAVE LoRA ADAPTER
# ======================================================
trainer.model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("\n✅ Training finished. Adapter saved to:", OUT_DIR)
