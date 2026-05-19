FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV PYVISTA_OFF_SCREEN=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /npet2

COPY pyproject.toml .
COPY libnpet/ libnpet/

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir .

ENV NPET2_ROOT=/data
ENV NPET2_RUNS_ROOT=/data/runs
ENV NPET2_CACHE_ROOT=/data/cache
ENV NPET2_RIBOXYZ_API_URL=https://api.ribosome.xyz

VOLUME ["/data"]

ENTRYPOINT ["npet2"]
CMD ["--help"]



