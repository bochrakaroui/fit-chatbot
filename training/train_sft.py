"""Fine-tune an instruction model with QLoRA and assistant-only loss."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path


def load_dependencies():
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install the ML dependencies: pip install -e '.[ml]'") from exc
    return (
        torch,
        load_dataset,
        LoraConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        SFTConfig,
        SFTTrainer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--model", help="Override model_name from the configuration")
    parser.add_argument("--smoke-test", action="store_true", help="Train only eight steps")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.model:
        config["model_name"] = args.model

    (
        torch,
        load_dataset,
        LoraConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        SFTConfig,
        SFTTrainer,
    ) = load_dependencies()

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    use_4bit = bool(config["use_4bit"] and use_cuda and platform.system() != "Windows")
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        device_map="auto" if use_cuda else None,
        torch_dtype=dtype if use_cuda else torch.float32,
        quantization_config=quantization,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={
            "train": config["train_file"],
            "validation": config["validation_file"],
        },
    )
    lora = LoraConfig(
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=float(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        max_length=int(config["max_length"]),
        assistant_only_loss=True,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bool(use_cuda and torch.cuda.is_bf16_supported()),
        fp16=bool(use_cuda and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=use_cuda,
        report_to="none",
        seed=seed,
        max_steps=8 if args.smoke_test else -1,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    Path(config["output_dir"], "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
