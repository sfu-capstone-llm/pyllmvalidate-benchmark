FROM python:3.8.20-slim-bullseye

# Install dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /workspace

# Default to bash for interactive use
CMD ["/bin/bash"]

