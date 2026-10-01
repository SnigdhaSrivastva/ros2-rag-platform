from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import get_settings
from embedding import EmbeddingService
from logging_config import configure_logging
from text_processing import build_prompt, collect_contexts

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AppState:
    tokenizer = None
    model = None
    qdrant = None
    embeddings = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info('Loading runtime dependencies...')
    AppState.tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_model_name,
        token=settings.hf_token,
    )
    AppState.model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.hf_model_name,
        token=settings.hf_token,
    )
    AppState.qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    AppState.embeddings = EmbeddingService(settings.embedding_model_name)
    logger.info('Runtime dependencies loaded successfully.')
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get('/healthz')
def healthz() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/ask')
def ask_question(request: QuestionRequest):
    try:
        query_vector = AppState.embeddings.encode(request.question)
        hits = AppState.qdrant.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=settings.retrieve_top_k,
        )

        if not hits:
            raise HTTPException(status_code=404, detail='No relevant context found')

        contexts, sources = collect_contexts(hit.payload for hit in hits)

        if not contexts:
            raise HTTPException(status_code=404, detail='Retrieved context is empty')

        prompt = build_prompt(request.question, contexts)[: settings.max_context_chars]
        tokens = AppState.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        output_ids = AppState.model.generate(**tokens, max_new_tokens=220, num_beams=4)
        answer = AppState.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if not answer:
            raise HTTPException(status_code=500, detail='Model generated empty response')

        return {
            'question': request.question,
            'answer': answer,
            'sources': sources,
            'contexts_used': len(contexts),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Unexpected failure during /ask')
        raise HTTPException(status_code=500, detail=f'Failed to generate answer: {exc}')
