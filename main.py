from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import json


def preprocess_emails(parsed_json_file):
    """
    Reads the parsed JSON file and preprocesses it into a single "text" field.
    """
    with open(parsed_json_file, "r", encoding="utf-8") as f:
        emails = json.load(f)
    
    # Combine subject and body for training
    for email in emails:
        if email["subject"] and email["body"]:  # Ensure valid data
            text = f"Subject: {email['subject']}\n\n{email['body']}"
            yield {"text": text}  # Use generator to avoid loading all data at once


def train_model(parsed_json_file):
    # Preprocess the parsed JSON file using streaming
    print("Loading and preprocessing emails...")
    dataset = load_dataset("json", data_files=parsed_json_file, split="train")  # Load as a streaming dataset

    # Load GPT-J tokenizer and model
    model_name = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Split into training and evaluation sets
    print("Splitting dataset...")
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        num_train_epochs=4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        save_total_limit=2,
        gradient_accumulation_steps=8,
        logging_dir="./logs",
        logging_steps=10,
        ddp_find_unused_parameters=False,  # Disable unused parameter check for DDP
        dataloader_num_workers=2,  # Reduce workers to save memory
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save the final model and tokenizer
    print("Saving model...")
    model.save_pretrained("./results")
    tokenizer.save_pretrained("./results")
    print("Training complete.")


if __name__ == "__main__":
    parsed_json_file = "enron_emails.json"  # File generated by your parsing script
    train_model(parsed_json_file)
