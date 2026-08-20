# winnex-ai-normalize — input normalization for the Madhava engine.
# Base: Python 3.12-slim. Consumes winnex-ai-normalize + winnex-madhava from PyPI.
FROM python:3.12-slim

WORKDIR /app

# The package is installed from PyPI (the image just runs the API).
RUN pip install --no-cache-dir winnex-ai-normalize[all]

# Provider registry persistence (mounted as a volume in production).
ENV WINNEX_AI_NORMALIZE_PROVIDERS_FILE=/var/lib/winnex-ai-normalize/providers.json
RUN mkdir -p /var/lib/winnex-ai-normalize

EXPOSE 8102

CMD ["python", "-c", "import uvicorn; uvicorn.run('winnex_ai_normalize.api.server:app', host='0.0.0.0', port=8102)"]
