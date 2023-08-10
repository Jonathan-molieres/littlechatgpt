FROM python:3.10.8
ARG PYTHON_ENV
ENV PYTHON_ENV=${PYTHON_ENV:-"production"}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    TERM=xterm-256color \
    DISABLE_COLLECTSTATIC=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.2.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN set -xe \
    && packages=' \
    ca-certificates \
    python3-lxml \
    ffmpeg \
    libsqlite3-mod-spatialite \
    ' \
    && buildPackages=' \
    gcc \
    curl \
    build-essential \
    libpq-dev \
    tk-dev \
    uuid-dev \
    binutils \
    libproj-dev \
    gdal-bin \
    pkg-config libgoogle-perftools-dev \
    libsentencepiece-dev \
    ' \
    && apt-get -qq -y update \
    && apt-get install -y $packages \
    && apt-get install -y $buildPackages --no-install-recommends \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
    && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb \
    && wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" >> /etc/apt/sources.list.d/c
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
COPY . .
# Installer les dépendances Python
RUN if [ "$PYTHON_ENV" = "production" ]; then poetry install --only main; fi

RUN if [ "$PYTHON_ENV" = "production" ]; then set -xe \
    && apt-get -qq -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    ; fi
ENTRYPOINT [ "/bin/bash", "/app/entrypoint" ]

CMD [ "fastapi" ]