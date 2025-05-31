# Vingi Setup Guide

## Overview

This guide walks you through setting up the Vingi development and runtime environment. The setup process is designed to be automated where possible while maintaining security and privacy standards.

## System Requirements

### Hardware Requirements

**Minimum Configuration:**
- **Processor**: Apple M1 or Intel x86_64 with AVX2 support
- **Memory**: 8GB RAM (16GB recommended for ML model training)
- **Storage**: 50GB available space (SSD recommended)
- **Graphics**: Integrated graphics sufficient (dedicated GPU optional for faster ML inference)

**Recommended Configuration:**
- **Processor**: Apple M2 Pro/Max or Intel i7/i9
- **Memory**: 32GB RAM for optimal performance
- **Storage**: 100GB available space on SSD
- **Graphics**: Apple GPU or NVIDIA GPU with CUDA support (for accelerated ML training)

### Software Requirements

**Operating System:**
- **macOS**: 14.0 (Sonoma) or later
- **iOS**: 17.0 or later (for mobile companion app)
- **Development**: Xcode 15.0+ required for iOS/macOS development

**Runtime Environment:**
- **Python**: 3.11 or later (3.12 recommended)
- **Swift**: 5.9 or later
- **Node.js**: 18.0+ (for development tools)
- **Git**: 2.40+ with LFS support

## Environment Setup

### 1. Development Tools Installation

#### Install Xcode and Command Line Tools

```bash
# Install Xcode from App Store or Apple Developer portal
# Then install command line tools
xcode-select --install

# Verify installation
swift --version
clang --version
```

#### Install Homebrew (Package Manager)

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH (for Apple Silicon Macs)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# Verify installation
brew --version
```

#### Install Core Dependencies

```bash
# Essential development tools
brew install git git-lfs python@3.12 node pipenv poetry

# Database and storage
brew install neo4j sqlite redis

# Security tools
brew install gnupg pinentry-mac

# Development utilities
brew install jq yq htop tree fd ripgrep

# Optional: GPU acceleration (if supported)
brew install cmake pkg-config
```

### 2. Python Environment Setup

#### Configure Python Environment

```bash
# Set Python 3.12 as default
brew link python@3.12

# Verify Python installation
python3 --version
pip3 --version

# Install pipenv for virtual environment management
pip3 install --upgrade pipenv poetry

# Install development tools
pip3 install pre-commit black flake8 mypy pytest
```

#### Create Virtual Environment

```bash
# Navigate to project directory (after cloning)
cd vingi

# Create virtual environment with specific Python version
pipenv install --python 3.12

# Activate virtual environment
pipenv shell

# Verify environment
which python
python --version
```

### 3. Swift Package Manager Setup

#### Configure Swift Environment

```bash
# Verify Swift Package Manager
swift package --version

# Set up Swift tools version (create in project root)
echo 'swift-tools-version:5.9' > Package.swift

# Configure Swift linting
brew install swiftlint swiftformat

# Verify Swift tools
swiftlint version
swiftformat --version
```

### 4. Database Setup

#### Neo4j Configuration

```bash
# Start Neo4j service
brew services start neo4j

# Configure Neo4j for Vingi
neo4j-admin set-initial-password vingi_secure_password

# Create Vingi database
cypher-shell -u neo4j -p vingi_secure_password "CREATE DATABASE vingi"

# Verify database connection
cypher-shell -u neo4j -p vingi_secure_password -d vingi "RETURN 'Database Connected' as status"
```

#### SQLite Setup

```bash
# SQLite is included with macOS, verify installation
sqlite3 --version

# Create application data directory
mkdir -p ~/Library/Application\ Support/Vingi/data

# Initialize SQLite databases
sqlite3 ~/Library/Application\ Support/Vingi/data/cache.db "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);"
```

### 5. Security Configuration

#### GPG Key Setup

```bash
# Generate GPG key for secure data encryption
gpg --full-generate-key
# Choose: RSA and RSA, 4096 bits, no expiration
# Use email: your-email@domain.com

# Configure pinentry for macOS
echo "pinentry-program /opt/homebrew/bin/pinentry-mac" >> ~/.gnupg/gpg-agent.conf

# Restart GPG agent
gpgconf --kill gpg-agent
```

#### SSH Key Configuration

```bash
# Generate SSH key for repository access
ssh-keygen -t ed25519 -C "your-email@domain.com" -f ~/.ssh/vingi_ed25519

# Add to SSH agent
ssh-add ~/.ssh/vingi_ed25519

# Copy public key to clipboard
pbcopy < ~/.ssh/vingi_ed25519.pub
# Add to your Git provider (GitHub, GitLab, etc.)
```

## Project Initialization

### 1. Repository Setup

#### Clone Repository

```bash
# Clone the repository
git clone git@github.com:yourusername/vingi.git
cd vingi

# Initialize Git LFS for large files
git lfs install
git lfs track "*.model" "*.pkl" "*.bin" "*.safetensors"

# Verify repository structure
tree -L 3 -I '__pycache__|*.pyc|node_modules'
```

#### Configure Git Hooks

```bash
# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Run initial hook check
pre-commit run --all-files
```

### 2. Dependency Installation

#### Install Python Dependencies

```bash
# Activate virtual environment
pipenv shell

# Install core dependencies
pipenv install torch torchvision transformers
pipenv install scikit-learn numpy pandas
pipenv install neo4j-driver aiohttp fastapi
pipenv install cryptography keyring
pipenv install pytest pytest-asyncio pytest-cov

# Install development dependencies
pipenv install --dev black flake8 mypy
pipenv install --dev jupyter notebook ipython
pipenv install --dev sphinx sphinx-rtd-theme

# Verify installation
pip list | grep -E "(torch|transformers|scikit|neo4j)"
```

#### Install Swift Dependencies

```bash
# Update Swift package dependencies
swift package update

# Resolve dependencies
swift package resolve

# Build dependencies
swift build
```

### 3. Configuration Setup

#### Create Configuration Files

```bash
# Copy configuration template
cp config/config.template.yml config/development.yml

# Generate secure encryption key
python3 -c "
import secrets
import base64
key = secrets.token_bytes(32)
print(f'encryption_key: {base64.b64encode(key).decode()}')" >> config/development.yml

# Set up logging configuration
cp config/logging.yml config/logging.development.yml
```

#### Configure Environment Variables

```bash
# Create environment file
cat > .env << EOF
# Development Environment
VINGI_ENV=development
VINGI_CONFIG_PATH=config/development.yml
VINGI_DATA_PATH=data/
VINGI_LOG_LEVEL=DEBUG

# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=vingi_secure_password
NEO4J_DATABASE=vingi

# Security Configuration
VINGI_ENCRYPTION_KEY_ID=development
VINGI_GPG_KEY_ID=$(gpg --list-secret-keys --keyid-format=long | grep 'sec ' | awk '{print $2}' | cut -d'/' -f2 | head -1)

# API Configuration
VINGI_API_HOST=127.0.0.1
VINGI_API_PORT=8000
VINGI_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# ML Model Configuration
HUGGINGFACE_CACHE_DIR=data/models/huggingface
TORCH_HOME=data/models/torch
TRANSFORMERS_CACHE=data/models/transformers
EOF

# Load environment variables
source .env
```

### 4. Initial Data Setup

#### Download Pre-trained Models

```bash
# Create model directories
mkdir -p data/models/{embeddings,classifiers,custom}
mkdir -p data/models/{huggingface,torch,transformers}

# Download sentence transformer model
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('data/models/embeddings/sentence-transformer')
print('Sentence transformer model downloaded successfully')
"

# Download classification model
python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer.save_pretrained('data/models/classifiers/distilbert-tokenizer')
model.save_pretrained('data/models/classifiers/distilbert-model')
print('Classification model downloaded successfully')
"
```

#### Initialize Databases

```bash
# Run database initialization script
./scripts/setup/setup_database.sh

# Verify database setup
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'vingi_secure_password'))
with driver.session(database='vingi') as session:
    result = session.run('RETURN 1 as test')
    print(f'Neo4j connection: {result.single()[0] == 1}')
driver.close()
"
```

### 5. Build and Test

#### Build Swift Components

```bash
# Build in debug mode
swift build

# Build in release mode (for production)
swift build -c release

# Run Swift tests
swift test

# Build Xcode project (if using Xcode)
xcodebuild -scheme Vingi -destination 'platform=macOS' build
```

#### Test Python Components

```bash
# Activate virtual environment
pipenv shell

# Run Python tests
pytest src/python/tests/ -v --cov=src/python/vingi

# Run specific test categories
pytest src/python/tests/test_core/ -v
pytest src/python/tests/test_nlp/ -v

# Generate coverage report
pytest --cov=src/python/vingi --cov-report=html
open htmlcov/index.html  # View coverage report
```

#### Integration Tests

```bash
# Run integration tests
python3 -m pytest tests/integration/ -v

# Test Swift-Python bridge
python3 tests/integration/test_swift_python_bridge.py

# Performance benchmarks
python3 tests/performance/test_latency.py
```

## Development Workflow Setup

### 1. IDE Configuration

#### Visual Studio Code Setup

```bash
# Install VS Code extensions
code --install-extension ms-python.python
code --install-extension swift-server.swift
code --install-extension ms-toolsai.jupyter
code --install-extension ms-vscode.vscode-json

# Configure VS Code settings
mkdir -p .vscode
cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "$(pipenv --venv)/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "swift.path": "/usr/bin/swift",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/build": true,
        "**/.build": true
    }
}
EOF
```

#### Xcode Project Setup

```bash
# Generate Xcode project
swift package generate-xcodeproj

# Open in Xcode
open Vingi.xcodeproj
```

### 2. Debugging Configuration

#### Python Debugging

```bash
# Install debugging tools
pipenv install --dev debugpy ipdb

# Configure VS Code debugging
cat > .vscode/launch.json << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Vingi Core",
            "type": "python",
            "request": "launch",
            "program": "src/python/vingi/__main__.py",
            "console": "integratedTerminal",
            "envFile": ".env"
        },
        {
            "name": "Python: Test Suite",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "envFile": ".env"
        }
    ]
}
EOF
```

#### Swift Debugging

```bash
# Enable debug symbols in Package.swift
cat >> Package.swift << EOF
// Debug configuration
#if DEBUG
    .unsafeFlags(["-Xswiftc", "-g"])
#endif
EOF
```

### 3. Documentation Setup

#### Generate Documentation

```bash
# Python documentation
cd src/python
sphinx-quickstart docs/
sphinx-apidoc -o docs/source/ vingi/
make -C docs/ html

# Swift documentation
swift package generate-documentation
```

### 4. Continuous Integration Setup

#### GitHub Actions Configuration

```bash
# Create CI/CD workflow
mkdir -p .github/workflows
cat > .github/workflows/ci.yml << EOF
name: Vingi CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-python:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        pip install pipenv
        pipenv install --dev
    - name: Run tests
      run: pipenv run pytest

  test-swift:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build Swift
      run: swift build
    - name: Run Swift tests
      run: swift test
EOF
```

## Production Deployment Setup

### 1. Release Configuration

#### Production Environment

```bash
# Create production configuration
cp config/development.yml config/production.yml

# Update production settings
sed -i '' 's/log_level: DEBUG/log_level: INFO/' config/production.yml
sed -i '' 's/debug: true/debug: false/' config/production.yml

# Generate production encryption keys
python3 -c "
import secrets
import base64
key = secrets.token_bytes(32)
print(f'Production encryption key: {base64.b64encode(key).decode()}')
" # Store securely in keychain
```

#### Build Release Binary

```bash
# Build optimized release
swift build -c release --arch arm64 --arch x86_64

# Create application bundle
./scripts/deployment/create_installer.sh
```

### 2. Security Hardening

#### Code Signing

```bash
# Create development certificate
security create-keychain -p "" vingi-dev.keychain
security import developer_certificate.p12 -k vingi-dev.keychain

# Sign application
codesign --force --verify --verbose --sign "Developer ID Application: Your Name" \
    .build/release/VingiCore
```

#### Notarization

```bash
# Submit for notarization
xcrun notarytool submit vingi-installer.dmg \
    --apple-id your-email@domain.com \
    --password your-app-specific-password \
    --team-id YOUR_TEAM_ID
```

## Troubleshooting

### Common Issues

#### Python Environment Issues

```bash
# Reset Python environment
pipenv --rm
pipenv install

# Fix SSL certificate issues
/Applications/Python\ 3.12/Install\ Certificates.command
```

#### Swift Build Issues

```bash
# Clean build directory
swift package clean
rm -rf .build/

# Reset package cache
swift package reset
swift package update
```

#### Database Connection Issues

```bash
# Restart Neo4j
brew services restart neo4j

# Check Neo4j status
brew services list | grep neo4j

# Reset database
neo4j-admin database delete vingi --force
cypher-shell -u neo4j -p vingi_secure_password "CREATE DATABASE vingi"
```

### Performance Optimization

#### Memory Usage

```bash
# Monitor memory usage
./tools/monitoring/resource_tracker.py --memory

# Optimize Python memory
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=4
```

#### Disk Space

```bash
# Clean model cache
./scripts/maintenance/cleanup_cache.sh

# Optimize database
cypher-shell -u neo4j -p vingi_secure_password -d vingi "CALL db.stats.clear()"
```

This setup guide ensures a robust, secure, and maintainable development environment for the Vingi personal AI assistant project.
