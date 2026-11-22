# Backend image (API, worker and one-off commands) for the examiner stack (docs/decisions/0026 §6).
#
#   VARIANT=gpu  CUDA 12.8 base, llama.cpp built with CUDA, torch cu128 (needs an NVIDIA GPU and the
#                NVIDIA Container Toolkit at run time; Blackwell sm_120 included)
#   VARIANT=cpu  plain Ubuntu, llama.cpp CPU build, torch CPU wheels (runs anywhere, slowly)
#
# Build context: the repository root (see docker/compose.yml).

ARG VARIANT=cpu

# ---- base images per variant ---------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS build-gpu
FROM ubuntu:24.04 AS build-cpu
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS run-gpu
FROM ubuntu:24.04 AS run-cpu

# ---- llama-cpp-python wheel (compiled once, in the build stage) ------------------------------
FROM build-${VARIANT} AS llama
ARG VARIANT
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# GPU: CUDA kernels for Turing to Blackwell. CPU: a portable build (no -march=native), so the
# image runs on any x86-64 machine with AVX2.
RUN if [ "$VARIANT" = "gpu" ]; then \
        export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;120"; \
    else \
        export CMAKE_ARGS="-DGGML_NATIVE=off -DGGML_AVX2=on"; \
    fi \
    && pip wheel --no-deps -w /wheels llama-cpp-python==0.3.35

# ---- runtime -------------------------------------------------------------------------------------
FROM run-${VARIANT} AS runtime
ARG VARIANT
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv ffmpeg libsndfile1 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY --from=llama /wheels /wheels
COPY backend/requirements.txt /tmp/requirements.txt
# The CPU variant swaps the CUDA torch wheels for the CPU ones; everything else is identical.
RUN if [ "$VARIANT" = "cpu" ]; then \
        sed -i -e 's/+cu128/+cpu/' -e 's#whl/cu128#whl/cpu#' /tmp/requirements.txt; \
    fi \
    && pip install /wheels/llama_cpp_python-*.whl \
    && pip install -r /tmp/requirements.txt \
    && rm -rf /wheels /tmp/requirements.txt

# ctranslate2 (faster-whisper) loads the CUDA 12 runtime from the nvidia-* pip wheels
# (docs/decisions/0007); harmless on the CPU variant.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib \
    CUDA_CACHE_MAXSIZE=4294967296 \
    HF_HOME=/cache/huggingface

WORKDIR /app/backend
COPY backend/ /app/backend/
COPY docker/init_app.py docker/fetch_models.py /app/docker/
# backend/.env is never copied: configuration comes from the compose environment.
RUN rm -f /app/backend/.env

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
