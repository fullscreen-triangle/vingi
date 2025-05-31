# Vingi Project Makefile
# Build, test, and manage the Vingi personal AI assistant

# Configuration
PROJECT_NAME := Vingi
PYTHON_VERSION := 3.12
SWIFT_BUILD_CONFIG := debug
PLATFORM := macos

# Directories
SRC_DIR := src
DOCS_DIR := docs
TESTS_DIR := tests
BUILD_DIR := .build
DIST_DIR := dist

# Python virtual environment
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

.PHONY: help setup clean build test lint format docs install uninstall

# Default target
help: ## Show this help message
	@echo "$(BLUE)Vingi Project - Personal AI Assistant$(NC)"
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup and Installation
setup: setup-python setup-swift setup-git ## Setup complete development environment
	@echo "$(GREEN)✓ Development environment setup complete$(NC)"

setup-python: ## Setup Python virtual environment and dependencies
	@echo "$(BLUE)Setting up Python environment...$(NC)"
	python$(PYTHON_VERSION) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,test,docs]"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Python environment ready$(NC)"

setup-swift: ## Setup Swift package dependencies
	@echo "$(BLUE)Setting up Swift dependencies...$(NC)"
	swift package update
	swift package resolve
	@echo "$(GREEN)✓ Swift dependencies ready$(NC)"

setup-git: ## Setup git hooks and configuration
	@echo "$(BLUE)Setting up git hooks...$(NC)"
	$(VENV_DIR)/bin/pre-commit install
	$(VENV_DIR)/bin/pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✓ Git hooks installed$(NC)"

# Building
build: build-swift build-python ## Build all components
	@echo "$(GREEN)✓ Build complete$(NC)"

build-swift: ## Build Swift components
	@echo "$(BLUE)Building Swift components...$(NC)"
	swift build -c $(SWIFT_BUILD_CONFIG)

build-swift-release: ## Build Swift components in release mode
	@echo "$(BLUE)Building Swift components (release)...$(NC)"
	swift build -c release

build-python: $(VENV_DIR) ## Build Python components
	@echo "$(BLUE)Building Python components...$(NC)"
	$(PYTHON) -m build

build-docs: $(VENV_DIR) ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd $(SRC_DIR)/python && $(PYTHON) -m sphinx -b html docs/ $(DOCS_DIR)/_build/html/

# Testing
test: test-swift test-python ## Run all tests
	@echo "$(GREEN)✓ All tests completed$(NC)"

test-swift: ## Run Swift tests
	@echo "$(BLUE)Running Swift tests...$(NC)"
	swift test --enable-code-coverage

test-python: $(VENV_DIR) ## Run Python tests
	@echo "$(BLUE)Running Python tests...$(NC)"
	$(PYTHON) -m pytest src/python/tests/ -v --cov=src/python/vingi --cov-report=html

test-integration: $(VENV_DIR) ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/integration/ -v

test-performance: $(VENV_DIR) ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) -m pytest tests/performance/ -v

# Code Quality
lint: lint-swift lint-python ## Run all linters
	@echo "$(GREEN)✓ Linting complete$(NC)"

lint-swift: ## Lint Swift code
	@echo "$(BLUE)Linting Swift code...$(NC)"
	swiftlint lint --strict

lint-python: $(VENV_DIR) ## Lint Python code
	@echo "$(BLUE)Linting Python code...$(NC)"
	$(PYTHON) -m flake8 src/python/
	$(PYTHON) -m mypy src/python/vingi/

format: format-swift format-python ## Format all code
	@echo "$(GREEN)✓ Code formatting complete$(NC)"

format-swift: ## Format Swift code
	@echo "$(BLUE)Formatting Swift code...$(NC)"
	swiftformat src/core/ --swiftversion 5.9

format-python: $(VENV_DIR) ## Format Python code
	@echo "$(BLUE)Formatting Python code...$(NC)"
	$(PYTHON) -m black src/python/
	$(PYTHON) -m isort src/python/

# Development Tools
dev-run: $(VENV_DIR) ## Run development server
	@echo "$(BLUE)Starting development server...$(NC)"
	$(PYTHON) -m vingi.cli.main --config config/development.yml

dev-python-bridge: $(VENV_DIR) ## Start Python bridge server for development
	@echo "$(BLUE)Starting Python bridge server...$(NC)"
	$(PYTHON) -m vingi.bridge.server --host 127.0.0.1 --port 8765

check-deps: ## Check for outdated dependencies
	@echo "$(BLUE)Checking Swift dependencies...$(NC)"
	swift package show-dependencies
	@echo "$(BLUE)Checking Python dependencies...$(NC)"
	$(PIP) list --outdated

# Database Management
db-setup: ## Setup local databases
	@echo "$(BLUE)Setting up databases...$(NC)"
	./scripts/setup/setup_database.sh

db-reset: ## Reset databases (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will delete all database data$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		./scripts/setup/reset_databases.sh; \
	else \
		echo ""; \
		echo "Cancelled"; \
	fi

# Deployment
package: clean build-swift-release build-python ## Create distribution packages
	@echo "$(BLUE)Creating distribution packages...$(NC)"
	mkdir -p $(DIST_DIR)
	# Create macOS app bundle
	./scripts/deployment/create_installer.sh
	# Create Python wheel
	$(PYTHON) -m build --wheel --outdir $(DIST_DIR)
	@echo "$(GREEN)✓ Packages created in $(DIST_DIR)/$(NC)"

install-cli: $(VENV_DIR) ## Install CLI tools globally
	@echo "$(BLUE)Installing CLI tools...$(NC)"
	$(PIP) install -e .
	ln -sf $(PWD)/$(VENV_DIR)/bin/vingi /usr/local/bin/vingi
	@echo "$(GREEN)✓ CLI tools installed$(NC)"

# Maintenance
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)
	rm -rf src/python/build/
	rm -rf src/python/dist/
	rm -rf src/python/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	swift package clean
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(BLUE)Deep cleaning...$(NC)"
	rm -rf $(VENV_DIR)
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	@echo "$(GREEN)✓ Deep cleanup complete$(NC)"

security-scan: $(VENV_DIR) ## Run security scans
	@echo "$(BLUE)Running security scans...$(NC)"
	$(PIP) install safety bandit
	$(PYTHON) -m safety check
	$(PYTHON) -m bandit -r src/python/vingi/

update-deps: $(VENV_DIR) ## Update all dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	swift package update
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements.txt

# Documentation
docs-serve: build-docs ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	cd $(DOCS_DIR)/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build
	rm -rf $(DOCS_DIR)/_build/

# Utility targets
$(VENV_DIR):
	@echo "$(YELLOW)Virtual environment not found. Run 'make setup-python' first.$(NC)"
	@exit 1

doctor: ## Run system diagnostic
	@echo "$(BLUE)Running system diagnostic...$(NC)"
	@echo "Python version: $$(python --version 2>&1 || echo 'Not found')"
	@echo "Swift version: $$(swift --version 2>&1 | head -n1 || echo 'Not found')"
	@echo "Git version: $$(git --version 2>&1 || echo 'Not found')"
	@echo "Neo4j status: $$(brew services list | grep neo4j || echo 'Not installed')"
	@if [ -d "$(VENV_DIR)" ]; then echo "Virtual environment: ✓"; else echo "Virtual environment: ✗"; fi
	@if [ -f "$(BUILD_DIR)/debug/VingiCore" ]; then echo "Swift build: ✓"; else echo "Swift build: ✗"; fi

version: ## Show version information
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $$(grep version pyproject.toml | head -n1 | cut -d'"' -f2)"
	@echo "Python: $(PYTHON_VERSION)"
	@echo "Platform: $(PLATFORM)"

# Development shortcuts
dev: setup build test ## Full development setup, build, and test
	@echo "$(GREEN)✓ Development workflow complete$(NC)"

ci: lint test ## Continuous integration workflow
	@echo "$(GREEN)✓ CI workflow complete$(NC)"

release: clean test build-swift-release package ## Release workflow
	@echo "$(GREEN)✓ Release build complete$(NC)"
