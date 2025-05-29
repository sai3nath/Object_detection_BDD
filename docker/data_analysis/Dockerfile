# ---- Stage 1: Builder ----
FROM python:3.9-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY ./docker/data_analysis/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Stage 2: Final Application Image ----
FROM python:3.9-slim-bookworm AS final

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY ./src /app/src
COPY ./configs /app/configs

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "src/master_analyzer.py", "--config", "configs/docker_gt_config.yaml"]
