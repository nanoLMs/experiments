# Multi-stage build for optimized container size
FROM python:3.9-slim as base

# Inystem dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
FROM base as python-deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM base as final
COPY --from=python-deps /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash nanolm
USER nanolm
WORKDIR /home/nanolm

# Copy application files
COPY --chown=nanolm:nanolm app.py .
COPY --chown=nanolm:nanolm model/ ./model/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONPATH=/home/nanolm
ENV MODEL_PATH=/home/nanolm/model
ENV PORT=8080
ENV LOG_LEVEL=INFO

# Run application
CMD ["python", "app.py"]
