# ROS2 Production RAG Platform

Production-oriented Retrieval Augmented Generation stack for ROS2, Nav2, MoveIt2, and Gazebo knowledge retrieval.

## Components
- **ETL Pipeline**: deterministic crawler with MongoDB upsert semantics.
- **Featurization Pipeline**: chunked embedding generation and Qdrant upsert.
- **API Service**: FastAPI `/ask` endpoint with retrieval-grounded generation.

## Quick Start
1. Copy env template:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
3. Run ETL:
   ```bash
   python app/etl_pipeline.py
   ```
4. Build vector index:
   ```bash
   python app/featurization_pipeline.py
   ```
5. Run API:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Production Practices Included
- Environment-based configuration (`pydantic-settings`)
- No hardcoded secrets
- Retrieval payload consistency (`text`, `url`, `source`, `chunk_id`)
- Health endpoint (`/healthz`)
- Structured logging baseline

## Next Hardening Steps
- API authentication & tenant quotas
- CI/CD with tests + security scans
- Observability: metrics, tracing, dashboards
