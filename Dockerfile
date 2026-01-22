# Base Image
FROM python:3.12-slim

# Install AWS Lambda Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Configure uv:
# 1. Compile bytecode for faster startup
# 2. Use copy mode (safer in Docker than hardlinks)
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy project definition files
# IMPORTANT: You must have pyproject.toml and uv.lock in your build context
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync
# --frozen: strict sync using uv.lock (fails if lock is out of date)
# --no-dev: exclude development dependencies
# --no-install-project: install ONLY dependencies, not the app itself (allows caching deps before copying code)
RUN uv sync --frozen --no-dev --no-install-project

# update PATH to use the virtual environment created by uv
# This allows 'python' and 'uvicorn' to run from the venv automatically
ENV PATH="/app/.venv/bin:$PATH"

# Copy application files and models
COPY *.py .
COPY models/ ./models/

# Set environment variables for Lambda Adapter
ENV PORT=8080 

# Create logs directory and set permissions
RUN mkdir logs && chmod 777 logs
RUN chmod +r serve.py models/best_model.pkl models/preprocessor.b

# Expose ports
EXPOSE 8080
EXPOSE 8050

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Default Command
# Since PATH is updated, this uses the python inside .venv
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]