FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Trained artefacts are mounted or copied in; the service reports "degraded"
# on /health until a model is present rather than refusing to start.
COPY models ./models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "loadfc.service:app", "--host", "0.0.0.0", "--port", "8000"]
