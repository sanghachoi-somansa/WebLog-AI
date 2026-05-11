# 사내 Docker 미러: docker compose 빌드 시 --build-arg 또는 .env 의 PYTHON_BASE
ARG PYTHON_BASE=python:3.12-slim-bookworm
FROM ${PYTHON_BASE}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY core ./core
COPY .streamlit ./.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
