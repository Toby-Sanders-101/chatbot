import requests
from bs4 import BeautifulSoup
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor

base_url = 'https://arxiv.org/html/'
num_papers_to_try = 10000
log_file = 'crawled_urls.txt'
max_links_to_crawl = 1000
RETRY_WAIT_TIME = 10
MAX_RETRIES = 5

# Get a list of paper IDs from the arXiv CS recent papers index
def get_paper_ids_from_index_page():
    url = 'https://arxiv.org/list/cs.AR/recent'
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:129.0) Gecko/20100101 Firefox/129.0'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching the index page: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    paper_ids = []
    # Look for <a> tags inside <dt> that contain the paper ID in the href
    for entry in soup.find_all('dt'):
        a_tag = entry.find('a', href=re.compile(r'^/abs/\d{4}\.\d{5}$'))
        if a_tag:
            paper_id = a_tag.get('href').split('/abs/')[1]  # Extract ID from '/abs/2411.12444'
            paper_ids.append(paper_id)
    
    return paper_ids


def log_url(url):
    print(url)
    with open(log_file, 'a') as f:
        f.write(url + '\n')

def fetch(url):
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Will raise an exception for 4xx/5xx responses
            return response.text
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                print(f"Rate limit hit for {url}. Retrying in {RETRY_WAIT_TIME} seconds...")
                time.sleep(RETRY_WAIT_TIME)  # Wait before retrying
                retries += 1
            else:
                print(f"HTTPError fetching {url}: {e}")
                break
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            break
    return ""

def get_html_content(url):
    html = fetch(url)
    if html:
        soup = BeautifulSoup(html, 'lxml')
        content = soup.find_all(['p', 'h1', 'h2', 'h3', 'pre', 'code'])
        text = ' '.join(el.get_text() for el in content)
        return text.strip()
    return ""

def try_versions(paper_id):
    version = 1
    while version <= 5:
        versioned_id = f"{paper_id}v{version}"
        url = f"{base_url}{versioned_id}"
        print(f"Trying URL: {url}")
        text = get_html_content(url)
        if text:
            print(f"Found valid version {versioned_id}!")
            return text
        version += 1
    return ""

def crawl_and_collect_single(paper_id):
    print(f"Attempting to fetch paper with ID: {paper_id}")
    text = try_versions(paper_id)
    if text:
        log_url(f"https://arxiv.org/html/{paper_id}")
        return text
    return ""

def crawl_and_collect():
    collected_texts = []
    paper_ids = get_paper_ids_from_index_page()  # Get paper IDs from the index page
    print(f"Found {len(paper_ids)} paper IDs.")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        
        for paper_id in paper_ids[:num_papers_to_try]:  # Limit the number of papers to crawl
            futures.append(executor.submit(crawl_and_collect_single, paper_id))
        
        for future in futures:
            result = future.result()
            if result:
                collected_texts.append(result)
    
    return collected_texts

def save_to_jsonl(texts, filename='html_fine_tune_data.jsonl'):
    if not texts:
        print("No texts to save.")
        return
    
    with open(filename, 'a') as f:
        for text in texts:
            json.dump({"text": text}, f)
            f.write('\n')

def main():
    filename = 'html_fine_tune_data.jsonl'
    
    # Clear the file at the start to ensure it is empty
    with open(filename, 'w') as f:
        f.write("") 
    
    # Crawl and collect papers
    texts = crawl_and_collect()

    # Check if any papers were collected
    if texts:
        print(f"Collected {len(texts)} texts.")
        save_to_jsonl(texts)
    else:
        print("No valid papers found. Skipping training.")

if __name__ == "__main__":
    main()
