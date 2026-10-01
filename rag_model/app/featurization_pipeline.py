from __future__ import annotations

import logging
import uuid

from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from config import get_settings
from embedding import EmbeddingService
from logging_config import configure_logging
from text_processing import chunk_text


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    mongo = MongoClient(settings.mongo_uri)
    coll = mongo[settings.mongo_database][settings.mongo_collection]
    docs = list(coll.find())
    if not docs:
        logger.warning('No source documents found. Nothing to featurize.')
        return

    embedding = EmbeddingService(settings.embedding_model_name)
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    dim = len(embedding.encode('dimension check'))
    ensure_collection(qdrant, settings.qdrant_collection, dim)

    points: list[PointStruct] = []
    for doc in docs:
        text = (doc.get('text_content') or doc.get('content') or '').strip()
        for chunk in chunk_text(text):
            vector = embedding.encode(chunk.text)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        'text': chunk.text,
                        'source': doc.get('source', 'unknown'),
                        'url': doc.get('url', ''),
                        'chunk_id': chunk.chunk_id,
                    },
                )
            )

    if points:
        qdrant.upsert(collection_name=settings.qdrant_collection, points=points)
        logger.info('Upserted %s chunks to Qdrant collection %s', len(points), settings.qdrant_collection)


if __name__ == '__main__':
    main()
