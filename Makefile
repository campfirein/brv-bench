SHELL := /bin/bash
ENV_FILE := .env
-include ${ENV_FILE}

# ==============================================================================

define show_header
	@echo "============================================================"
	@echo $(1)
	@echo "============================================================"
endef

# ==============================================================================

all: clean format lint

# ==============================================================================

clean:
	$(call show_header, "Cleaning Source Code...")
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.py.log*" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -print -delete

# ==============================================================================

test:
	$(call show_header, "Testing...")
	pytest -v

# ==============================================================================

format:
	$(call show_header, "Formatting Source Code...")
	ruff format .

# ==============================================================================

lint:
	$(call show_header, "Linting Source Code...")
	ruff check --fix .

# ==============================================================================
