from __future__ import annotations

import logging
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, UpdateOne

from config import get_settings
from logging_config import configure_logging


def scrape(url: str) -> tuple[str, list[str]]:
    resp = requests.get(url, timeout=20, headers={'User-Agent': 'RAGBot/1.0'})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'lxml')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    links = [a.get('href') for a in soup.find_all('a', href=True)]
    return text, links


def crawl(seed: str, max_pages: int = 40) -> list[dict]:
    domain = urlparse(seed).netloc
    seen = set()
    queue = [seed]
    docs = []
    while queue and len(seen) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            text, links = scrape(url)
            docs.append({'url': url, 'text_content': text, 'source': domain, 'ingested_at': int(time.time())})
            for link in links:
                absolute = urljoin(url, link)
                if urlparse(absolute).netloc == domain and absolute not in seen:
                    queue.append(absolute)
        except Exception:
            continue
    return docs


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    sources = [
        'https://docs.ros.org/en/foxy/',
        'https://docs.nav2.org/',
        'https://moveit.picknik.ai/main/index.html',
        'https://gazebosim.org/docs/all/getstarted/',
    ]

    mongo = MongoClient(settings.mongo_uri)
    coll = mongo[settings.mongo_database][settings.mongo_collection]

    ops = []
    for source in sources:
        for doc in crawl(source):
            ops.append(
                UpdateOne(
                    {'url': doc['url']},
                    {'$set': doc},
                    upsert=True,
                )
            )
    if ops:
        result = coll.bulk_write(ops, ordered=False)
        logger.info('ETL complete. upserted=%s modified=%s', result.upserted_count, result.modified_count)


if __name__ == '__main__':
    main()
