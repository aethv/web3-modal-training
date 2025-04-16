# train.py
import os
import torch
import logging
import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
import json
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Web3 code generation model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to resume from")
    return parser.parse_args()

def train_model(config_path, resume=False, checkpoint_path=None):
    """Train the model with the specified configuration"""
    logger.info(f"Starting training with config from {config_path}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Initialize wandb for experiment tracking
    wandb.init(project="web3-app-generator", config=config)
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenized dataset
    dataset_path = os.path.join(config["data_dir"], "tokenized_dataset")
    logger.info(f"Loading tokenized dataset from {dataset_path}")
    tokenized_dataset = load_from_disk(dataset_path)
    
    # Load model and tokenizer
    if resume and checkpoint_path:
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        model_path = checkpoint_path
    else:
        model_path = os.path.join(config["model_dir"], "base_model")
    
    logger.info(f"Loading model and tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Determine model loading approach based on whether LoRA is used
    if config.get("use_lora", False):
        logger.info("Using LoRA for parameter-efficient fine-tuning")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config["lora_config"]["r"],
            lora_alpha=config["lora_config"]["lora_alpha"],
            lora_dropout=config["lora_config"]["lora_dropout"],
            bias=config["lora_config"]["bias"],
            task_type=TaskType.CAUSAL_LM
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.info("Using full model fine-tuning")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
        
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments from config
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=int(config["num_epochs"]),
        weight_decay=float(config["weight_decay"]),
        save_total_limit=int(config["save_total_limit"]),
        logging_steps=int(config["logging_steps"]),
        eval_steps=int(config["eval_steps"]),
        save_steps=int(config["save_steps"]),
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        fp16=device.type == "cuda",
        report_to="wandb",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training")
    if resume and not checkpoint_path:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save the final model
    final_model_path = os.path.join(config["output_dir"], "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Training complete. Final model saved to {final_model_path}")
    
    # Close wandb run
    wandb.finish()
    
    return final_model_path

if __name__ == "__main__":
    args = parse_args()
    final_model_path = train_model(args.config, args.resume, args.checkpoint)
    logger.info(f"Training script completed. Model available at: {final_model_path}")