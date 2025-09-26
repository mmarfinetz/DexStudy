.PHONY: help build run demo test clean shell logs stop test-local notebook-convert

# Default target
help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║     DEX Valuation Study - Claude Test Harness             ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available commands:"
	@echo "  make build      - Build Docker container with all dependencies"
	@echo "  make run        - Run Jupyter notebook server (interactive)"
	@echo "  make demo       - Run full Claude test harness demo (automated)"
	@echo "  make test       - Run generated test suite with coverage"
	@echo "  make shell      - Open shell in container for debugging"
	@echo "  make logs       - Show container logs"
	@echo "  make stop       - Stop all running containers"
	@echo "  make clean      - Clean generated files and caches"
	@echo "  make test-local - Run tests locally without Docker"
	@echo ""
	@echo "Quick start:"
	@echo "  1. export ANTHROPIC_API_KEY=your-key-here"
	@echo "  2. make build"
	@echo "  3. make demo"
	@echo ""

# Build Docker container
build:
	@echo "🔨 Building Docker container..."
	docker build -t dex-claude-harness:latest .
	@echo "✅ Build complete!"

# Run Jupyter notebook server
run: build
	@echo "🚀 Starting Jupyter notebook server..."
	@echo "📝 Access notebook at: http://localhost:8888"
	@echo "📁 Navigate to: /notebooks/claude_test_harness.ipynb"
	@echo ""
	docker run -it --rm \
		--name dex-harness \
		-p 8888:8888 \
		-v $(PWD):/app \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		dex-claude-harness:latest

# Run full demo (automated notebook execution)
demo: build
	@echo "🎯 Starting Claude Test Harness Demo..."
	@echo "This will:"
	@echo "  1. Generate comprehensive tests using Claude"
	@echo "  2. Run tests and capture failures"
	@echo "  3. Auto-generate patches for failures"
	@echo "  4. Apply patches and re-validate"
	@echo ""
	docker run -it --rm \
		-v $(PWD):/app \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		dex-claude-harness:latest \
		bash -c "cd /app && jupyter nbconvert --to notebook --execute \
		notebooks/claude_test_harness.ipynb \
		--output claude_test_harness_executed.ipynb \
		--ExecutePreprocessor.timeout=600"
	@echo ""
	@echo "✅ Demo complete! Check notebooks/claude_test_harness_executed.ipynb"
	@echo "📊 Generated tests are in: tests/generated/"
	@echo "🔧 Applied patches are in: patches/"

# Run test suite
test: build
	@echo "🧪 Running generated test suite..."
	docker run -it --rm \
		-v $(PWD):/app \
		-e PYTHONPATH=/app \
		dex-claude-harness:latest \
		pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "📊 Coverage report generated in htmlcov/"

# Open shell in container
shell: build
	@echo "🐚 Opening shell in container..."
	docker run -it --rm \
		-v $(PWD):/app \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e PYTHONPATH=/app \
		dex-claude-harness:latest \
		/bin/bash

# Show container logs
logs:
	docker logs -f dex-harness 2>/dev/null || echo "No running container found"

# Stop running containers
stop:
	@echo "🛑 Stopping containers..."
	docker stop dex-harness 2>/dev/null || true
	@echo "✅ Containers stopped"

# Run tests locally (without Docker)
test-local:
	@echo "🧪 Running tests locally..."
	pytest tests/ -v --cov=src --cov-report=term-missing

# Convert notebook to HTML for sharing
notebook-convert:
	@echo "📄 Converting notebook to HTML..."
	docker run -it --rm \
		-v $(PWD):/app \
		dex-claude-harness:latest \
		jupyter nbconvert --to html \
		notebooks/claude_test_harness.ipynb \
		--output claude_test_harness.html
	@echo "✅ HTML report created: notebooks/claude_test_harness.html"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf src/__pycache__ tests/__pycache__
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.backup" -delete
	rm -rf tests/generated/*
	rm -rf patches/*
	rm -f notebooks/claude_test_harness_executed.ipynb
	rm -f notebooks/claude_test_harness.html
	@echo "✅ Clean complete!"

# Install development dependencies locally
install-dev:
	pip install -r requirements-dev.txt
	@echo "✅ Development dependencies installed"

# Healthcheck
healthcheck:
	@docker run --rm dex-claude-harness:latest \
		python -c "import pandas, numpy, sklearn, pytest, anthropic; print('✅ All dependencies OK')" \
		2>/dev/null || echo "❌ Healthcheck failed - rebuild with 'make build'"

# Quick test of the harness without full demo
quick-test: build
	@echo "⚡ Running quick test..."
	docker run -it --rm \
		-v $(PWD):/app \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		dex-claude-harness:latest \
		python -c "from notebooks.claude_test_harness import ClaudeTestHarness; \
		harness = ClaudeTestHarness('../src/validation.py'); \
		print('✅ Harness initialized successfully')"

# Docker compose commands
compose-up:
	docker-compose up -d
	@echo "✅ Services started. Access Jupyter at http://localhost:8888"

compose-down:
	docker-compose down
	@echo "✅ Services stopped"

compose-logs:
	docker-compose logs -f

# Advanced: Run with GPU support (if available)
run-gpu: build
	@echo "🎮 Starting with GPU support..."
	docker run -it --rm \
		--gpus all \
		--name dex-harness \
		-p 8888:8888 \
		-v $(PWD):/app \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		dex-claude-harness:latest