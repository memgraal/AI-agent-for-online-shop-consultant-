FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /home/app/.cache/fastembed \
    && chown -R app:app /home/app/.cache

COPY faq_workbench/pyproject.toml ./pyproject.toml
COPY faq_workbench/alembic.ini ./alembic.ini
COPY faq_workbench/faq_workbench ./faq_workbench
COPY faq_workbench/migrations ./migrations

RUN python -m pip install --no-cache-dir .

USER app
EXPOSE 8010

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn faq_workbench.main:app --host 0.0.0.0 --port 8010"]
