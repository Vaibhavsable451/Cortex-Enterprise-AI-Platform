"""LoRA / QLoRA fine-tuning with Hugging Face + PEFT.

Examples:
  python -m deep_learning.lora_finetune --model Qwen/Qwen2.5-0.5B --dataset imdb --samples 500
  python -m deep_learning.lora_finetune --model Qwen/Qwen2.5-1.5B --qlora        # needs CUDA + bitsandbytes
Multi-GPU:  accelerate launch --use_fsdp ...   |   deepspeed ... (see README)
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    a.add_argument("--dataset", default="imdb")
    a.add_argument("--samples", type=int, default=500)
    a.add_argument("--qlora", action="store_true")
    a.add_argument("--out", default="outputs/lora")
    a.add_argument("--epochs", type=int, default=1)
    args = a.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    kw = {}
    if args.qlora:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    ds = load_dataset(args.dataset, split=f"train[:{args.samples}]")
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True,
                remove_columns=ds.column_names)
    cuda = torch.cuda.is_available()
    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=2,
            gradient_accumulation_steps=8, learning_rate=2e-4, warmup_ratio=0.05, lr_scheduler_type="cosine",
            bf16=cuda and torch.cuda.is_bf16_supported(), fp16=cuda and not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True, max_grad_norm=1.0, logging_steps=10, save_strategy="epoch",
            report_to="mlflow", optim="paged_adamw_8bit" if args.qlora else "adamw_torch"),
        train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
