import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import pandas as pd
from typing import List, Dict, Optional
import json
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishingDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: LlamaTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Create prompt template
        prompt = f"""Below is a potential phishing content. Analyze it and determine if it's phishing or legitimate.

Content: {text}

Analysis: This content is {"phishing" if label == 1 else "legitimate"}.

Reasoning: Let me explain why this is {"phishing" if label == 1 else "legitimate"}:
"""
        
        # Tokenize the prompt
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }

def prepare_data(data_path: str) -> tuple:
    
    df = pd.read_csv(data_path)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, val_df

def create_lora_config():
    """Create LoRA configuration."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def train_lora(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "data/phishing_dataset.csv",
    output_dir: str = "models/lora_phishing",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 100,
    fp16: bool = True
):
    """Main training function for LoRA fine-tuning."""
    
    # Initialize tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Create and apply LoRA configuration
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Prepare datasets
    train_df, val_df = prepare_data(data_path)
    
    train_dataset = PhishingDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = PhishingDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=fp16,
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Training completed. Model saved to {output_dir}")

def generate_phishing_analysis(
    model_path: str,
    text: str,
    max_length: int = 512
) -> str:
    """Generate phishing analysis for given text using the fine-tuned model."""
    
    # Load model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Create prompt
    prompt = f"""Below is a potential phishing content. Analyze it and determine if it's phishing or legitimate.

Content: {text}

Analysis:"""
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage
    train_lora(
        model_name="meta-llama/Llama-2-7b-hf",  # Replace with your model path
        data_path="data/phishing_dataset.csv",
        output_dir="models/lora_phishing",
        num_epochs=3
    ) 