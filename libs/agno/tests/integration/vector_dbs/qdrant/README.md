### Running test
```bash
cd libs/agno/tests/integration/vector_dbs/qdrant
docker compose -f docker-compose.local.yml up -d qdrant
pytest
```