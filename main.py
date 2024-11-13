import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
import json
import random
import time
from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np

base_url = 'https://arxiv.org/html/'

num_papers_to_try = 1000

max_links_to_crawl = 100

async def fetch(session, url, proxy=None):
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    try:
        async with session.get(url, headers=headers, proxy=proxy) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

async def findpaper(session, year, month, index):
    for version in range(1,8):
        url = f"https://arxiv.org/html/{year}{str(month).rjust(2, "0")}.{str(index).rjust(5, "0")}v{version}"
        html = await fetch(session, url)
        if html != "":
            print("hit",url)
            save_to_jsonl(html)
            return 1
    return 0

async def main():
    filename = 'html_fine_tune_data.jsonl'
    
    with open(filename, 'w') as f:
        f.write("") 

    async with aiohttp.ClientSession() as session:
        list = []
        for year in range(24,18,-1)
        x = np.sum(await asyncio.gather(*list))

    print(f"found {x} papers")

    train_gpt_model('html_fine_tune_data.jsonl')

def save_to_jsonl(texts, filename='html_fine_tune_data.jsonl'):
    with open(filename, 'a') as f:
        for text in texts:
            json.dump({"text": text}, f)
            f.write('\n')

def train_gpt_model(data_file):
    dataset = load_dataset('json', data_files=data_file)

    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
        tokenized['labels'] = tokenized['input_ids']
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        learning_rate=5e-3,
        per_device_train_batch_size=1,
        num_train_epochs=4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')

if __name__ == "__main__":
    asyncio.run(main())

