from datasets import load_dataset
from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
import torch

def train_gpt_model(data_file):
    # Load dataset from the provided JSONL file
    dataset = load_dataset('json', data_files=data_file)

    # Load pre-trained GPT-2 tokenizer and model
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set pad token as the end of sequence token for compatibility
    tokenizer.pad_token = tokenizer.eos_token

    # Function to tokenize the dataset
    def tokenize_function(examples):
        tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
        tokenized['labels'] = tokenized['input_ids']  # We use input_ids as labels for language modeling
        return tokenized

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Split the dataset into training and evaluation sets
    train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',              # Where to save the model and logs
        eval_strategy='epoch',               # Evaluate once per epoch
        learning_rate=5e-3,                  # Learning rate
        per_device_train_batch_size=1,       # Batch size per device for training
        num_train_epochs=4,                  # Number of training epochs
        weight_decay=0.01,                   # Weight decay for regularization
        fp16=torch.cuda.is_available(),      # Enable FP16 if CUDA is available
        logging_dir='./logs',                # Where to save logs
        logging_steps=10,                    # Log every 10 steps
        save_steps=100,                      # Save model every 100 steps
        save_total_limit=2,                  # Keep only the last 2 saved models
        gradient_accumulation_steps=8,       # Accumulate gradients for batch size > 1
    )

    # Create Trainer object
    trainer = Trainer(
        model=model,                         # The GPT-2 model
        args=training_args,                  # Training arguments
        train_dataset=train_dataset,         # Training dataset
        eval_dataset=eval_dataset,           # Evaluation dataset
    )

    # Start training
    trainer.train()

    # Save the model and tokenizer after training
    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')


if __name__ == "__main__":
    # Path to your JSONL file with training data
    data_file = 'html_fine_tune_data.jsonl'  # Update this if necessary
    train_gpt_model(data_file)
