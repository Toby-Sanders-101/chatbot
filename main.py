from transformers import GPTJForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def train_model(data_file):
    # Load dataset
    dataset = load_dataset("json", data_files=data_file)

    # Use GPT-J (6B) and corresponding tokenizer
    model_name = "EleutherAI/gpt-j-6B"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Handle padding compatibility
    model = GPTJForCausalLM.from_pretrained(model_name)

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_dataset = dataset['train'].map(tokenize_function, batched=True, remove_columns=["text"])
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save results
    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')

if __name__ == "__main__":
    data_file = "enron_emails.jsonl"
    train_model(data_file)
