# Multi-stage build for optimized image size
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install dependencies in virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Install Jupyter and testing tools
RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    ipykernel \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    pytest-timeout>=2.1.0 \
    pytest-xdist>=3.3.0 \
    pytest-benchmark>=4.0.0 \
    anthropic>=0.7.0 \
    nbconvert>=7.0.0

# Final stage
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/results/tables \
    /app/results/figures \
    /app/tests/generated \
    /app/patches \
    /app/notebooks

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV JUPYTER_ENABLE_LAB=yes

# Healthcheck - verify critical packages and services
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pandas, numpy, sklearn, pytest, anthropic; print('OK')" || exit 1

# Expose Jupyter port
EXPOSE 8888

# Configure Jupyter
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_origin = '*'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Default command - start Jupyter notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]