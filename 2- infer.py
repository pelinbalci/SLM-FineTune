import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "./outputs/qwen2_5_0_5b-capybara-lora"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

prompt = "Explain LoRA fine-tuning in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
