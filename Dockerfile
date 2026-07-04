FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir ".[server,data]"

# Non-root runtime user; artifacts/models live on mounted volumes.
RUN useradd --create-home titan \
    && mkdir -p /app/artifacts /app/data_cache /app/models_store \
    && chown -R titan:titan /app
USER titan

EXPOSE 8321

# Default: serve the dashboard over whatever artifacts are mounted.
# Run research with:  docker run ... titan validate --config configs/default.yaml
ENTRYPOINT ["titan"]
CMD ["dashboard", "--host", "0.0.0.0", "--port", "8321", "--artifacts", "artifacts"]
