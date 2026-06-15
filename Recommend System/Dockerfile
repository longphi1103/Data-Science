FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
RUN uv sync --frozen
RUN uv run playwright install --with-deps chromium

COPY configs ./configs
COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "absa_recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
