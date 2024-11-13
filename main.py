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

base_url = 'https://arxiv.org/html/'

num_papers_to_try = 1000

log_file = 'crawled_urls.txt'

max_links_to_crawl = 100

index = 0

def load_proxies():
    with open("proxies.txt", "r") as f:
        proxies = [line.strip() for line in f.readlines()]
    return proxies

def generate_random_paper_id():
    global index
    year = 23  # Fixing year to 2023
    month = random.randint(1, 12)
    paper_number = str(index).rjust(5, "0")
    index += 1
    return f"{year:02d}{month:02d}.{paper_number}"

def log_url(url):
    print(url)
    with open(log_file, 'a') as f:
        f.write(url + '\n')

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

async def get_html_content(session, url, proxy=None):
    html = await fetch(session, url, proxy)
    if html:
        soup = BeautifulSoup(html, 'lxml')
        content = soup.find_all(['p', 'h1', 'h2', 'h3', 'pre', 'code'])
        links = soup.find_all(['a'])
        text = ' '.join(el.get_text() for el in content)
        if re.search(r'\b(html|head|body|div|span|a|img|table|script|style)\b', text, re.I):
            return text.strip(), [a['href'] for a in soup.find_all('a', href=True)]
    return "", []

async def try_versions(session, paper_id, proxy=None):
    version = 1
    while version <= 5:
        versioned_id = f"{paper_id}v{version}"
        url = f"{base_url}{versioned_id}"
        print(f"Trying URL: {url}")
        text, _ = await get_html_content(session, url, proxy)
        if text:
            print(f"Found valid version {versioned_id}!")
            return text
        version += 1
    return ""

async def crawl_and_collect_single(paper_id, proxy=None):
    async with aiohttp.ClientSession() as session:
        print(f"Attempting to fetch paper with ID: {paper_id}")
        text = await try_versions(session, paper_id, proxy)
        if text:
            log_url(f"https://arxiv.org/html/{paper_id}")
            return text
    return ""

async def crawl_and_collect(proxies):
    collected_texts = []
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        for _ in range(num_papers_to_try):
            paper_id = generate_random_paper_id()
            proxy = random.choice(proxies)
            tasks.append(loop.create_task(crawl_and_collect_single(paper_id, proxy)))
        results = await asyncio.gather(*tasks)
        collected_texts = [result for result in results if result]
    return collected_texts

async def main():
    proxies = load_proxies()

    filename = 'html_fine_tune_data.jsonl'
    
    with open(filename, 'w') as f:
        f.write("") 
    
    texts = await crawl_and_collect(proxies)
    if texts:
        print(f"Collected {len(texts)} texts.")
        save_to_jsonl(texts)
    else:
        print("No valid papers found.")
    
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

