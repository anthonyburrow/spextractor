.PHONY: help venv install lint test test-cov clean

PYTHON := python
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
PYTEST_FLAGS := -sv

IN_VENV := $(shell python -c 'import sys; print(int(sys.prefix != sys.base_prefix))')

help:
	@echo "Available targets:"
	@echo "  make venv     - Create a virtual environment in .venv/"
	@echo "  make install  - Install the project (auto-detects venv)"
	@echo "  make lint     - Run Ruff linter"
	@echo "  make test     - Run pytest tests"
	@echo "  make test-cov - Run pytest with coverage reports"
	@echo "  make clean    - Remove build artifacts and venv"

venv:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	else \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Virtual environment created at $(VENV_DIR)"; \
		echo "Activate it with: source $(VENV_DIR)/bin/activate"; \
	fi

install:
	@if [ -d "$(VENV_DIR)" ] && [ "$(IN_VENV)" = "0" ]; then \
		echo "Virtual environment detected but not activated."; \
		echo "Installing in venv..."; \
		$(VENV_PIP) install --upgrade pip; \
		$(VENV_PIP) install -e .[dev] --force-reinstall --no-deps; \
		$(VENV_PIP) install -e .[dev]; \
	elif [ "$(IN_VENV)" = "1" ]; then \
		echo "Installing in active virtual environment..."; \
		pip install --upgrade pip; \
		pip install -e .[dev] --force-reinstall --no-deps; \
		pip install -e .[dev]; \
	else \
		echo "No virtual environment detected."; \
		echo "Installing in system/user Python..."; \
		pip install --upgrade pip; \
		pip install -e .[dev] --force-reinstall --no-deps; \
		pip install -e .[dev]; \
	fi
	@echo "Installation complete!"

lint:
	@if [ -d "$(VENV_DIR)" ] && [ "$(IN_VENV)" = "0" ]; then \
		echo "Running Ruff in venv..."; \
		$(VENV_PYTHON) -m ruff check --output-format=github --target-version=py310; \
	elif [ "$(IN_VENV)" = "1" ]; then \
		echo "Running Ruff in active virtual environment..."; \
		ruff check --output-format=github --target-version=py310; \
	else \
		echo "Running Ruff..."; \
		ruff check --output-format=github --target-version=py310; \
	fi

test:
	@if [ -d "$(VENV_DIR)" ] && [ "$(IN_VENV)" = "0" ]; then \
		echo "Running tests in venv..."; \
		$(VENV_PYTHON) -m pytest $(PYTEST_FLAGS); \
	elif [ "$(IN_VENV)" = "1" ]; then \
		echo "Running tests in active virtual environment..."; \
		pytest $(PYTEST_FLAGS); \
	else \
		echo "Running tests..."; \
		pytest $(PYTEST_FLAGS); \
	fi

test-cov:
	@if [ -d "$(VENV_DIR)" ] && [ "$(IN_VENV)" = "0" ]; then \
		echo "Running tests with coverage in venv..."; \
		$(VENV_PYTHON) -m pytest -s --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html; \
	elif [ "$(IN_VENV)" = "1" ]; then \
		echo "Running tests with coverage in active virtual environment..."; \
		pytest -s --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html; \
	else \
		echo "Running tests with coverage..."; \
		pytest -s --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html; \
	fi

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf $(VENV_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	@echo "Clean complete!"
