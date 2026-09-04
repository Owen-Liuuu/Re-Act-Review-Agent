# Build context is the repo root. Secrets stay out of the image.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir . openai \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data/runs /data/uploads \
    && chown -R app:app /data /app

USER app
VOLUME ["/data"]
EXPOSE 8080
# Keys and Basic Auth come from the environment / compose env_file. Not baked in.
CMD ["react-review", "serve", "--host", "0.0.0.0", "--port", "8080", "--out", "/data/runs", "--uploads", "/data/uploads", "--config", "/app/configs/config.example.yaml"]
