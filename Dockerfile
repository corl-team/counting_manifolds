FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    git \
    vim \
    wget \
    curl \
    python3-venv pipx \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:$PATH
RUN pipx install "poetry>=1.8.0"

WORKDIR /app
COPY pyproject.toml poetry.lock* ./

ENV POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_NO_ANSI=1 \
    PYTHONUNBUFFERED=1

RUN poetry env use /usr/bin/python3.12 && \
    poetry install --no-root \
    && rm -rf /root/.cache/pip /root/.cache/pypoetry

ENV PATH="/app/.venv/bin:$PATH"

### Install flash-attn from wheel
ARG FLASH_ATTN_VER=2.8.3

RUN python - <<'PY'
import sys, subprocess, os
import torch

ver = os.environ["FLASH_ATTN_VER"]
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])   # e.g. 2.8
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"

wheel = f"flash_attn-{ver}+cu12torch{torch_mm}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
url = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{ver}/{wheel}"

print("Installing:", url)
subprocess.check_call(["poetry", "run", "pip", "install", "--no-deps", url])
PY
