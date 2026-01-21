# Base Image
FROM python:3.12-slim

# Install AWS Lambda Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Set working directory
WORKDIR /app

# Install system dependencies for ML and Plotly rendering
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir uvicorn fastapi && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files and models
COPY *.py .
# Copy models directory
COPY models/ ./models/

# Set environment variables for Lambda Adapter
# Port 8080 for API, Port 8050 for Dash
ENV PORT=8080 

# Create logs directory
RUN mkdir logs && chmod 777 logs

# Set permissions for specific files
RUN chmod +r serve.py models/best_model.pkl models/preprocessor.b

# Expose ports
EXPOSE 8080
EXPOSE 8050

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Default Command (Overridden by Docker Compose or Lambda settings)
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]