# Vingi Installation Guide

## Overview

This guide provides step-by-step installation instructions for Vingi across different scenarios: end-user installation, developer installation, and custom deployment. The installation process prioritizes security, privacy, and ease of use.

## Installation Options

### Quick Installation (Recommended)

For most users, the automated installer provides the fastest and most reliable installation experience.

### Custom Installation

For developers or users requiring specific configurations, manual installation provides complete control over the setup process.

### Enterprise Installation

For organizational deployment with specific security or compliance requirements.

## Prerequisites Check

Before beginning installation, verify your system meets the minimum requirements:

```bash
# Run the system compatibility check
curl -fsSL https://raw.githubusercontent.com/yourusername/vingi/main/scripts/setup/check_prerequisites.sh | bash
```

**Expected Output:**
```
✓ Operating System: macOS 14.2 (Compatible)
✓ Processor: Apple M2 (Compatible)
✓ Memory: 16GB (Recommended)
✓ Storage: 150GB available (Sufficient)
✓ Python: Not installed (Will install)
✓ Swift: Not installed (Will install)
```

## Quick Installation

### Automated Installer (macOS)

The automated installer handles all dependencies, configurations, and initial setup:

```bash
# Download and run the official installer
curl -fsSL https://install.vingi.dev | bash

# Or with specific version
curl -fsSL https://install.vingi.dev | bash -s -- --version=1.0.0-beta
```

**Installation Process:**
1. **System Verification**: Checks compatibility and permissions
2. **Dependency Installation**: Installs required tools via Homebrew
3. **Application Download**: Downloads signed application bundle
4. **Security Verification**: Validates code signatures and checksums
5. **Initial Configuration**: Sets up user profile and privacy settings
6. **Launch**: Starts Vingi with setup wizard

**Installation Verification:**
```bash
# Verify installation
vingi --version
vingi doctor  # Run diagnostic check
```

### Manual Download (Alternative)

If you prefer manual installation:

1. **Download Application:**
   - Visit: [https://github.com/yourusername/vingi/releases](https://github.com/yourusername/vingi/releases)
   - Download: `Vingi-1.0.0-beta-macos.dmg`
   - Verify checksum: `shasum -a 256 Vingi-1.0.0-beta-macos.dmg`

2. **Install Application:**
   ```bash
   # Mount DMG
   hdiutil attach Vingi-1.0.0-beta-macos.dmg
   
   # Copy to Applications
   cp -R "/Volumes/Vingi/Vingi.app" /Applications/
   
   # Unmount DMG
   hdiutil detach "/Volumes/Vingi"
   ```

3. **Install CLI Tools:**
   ```bash
   # Install command-line interface
   sudo /Applications/Vingi.app/Contents/Resources/install-cli.sh
   ```

## Developer Installation

### Complete Development Environment

For developers who want to contribute to Vingi or customize functionality:

#### 1. Clone Repository

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/vingi.git
cd vingi

# Verify repository integrity
git verify-commit HEAD
```

#### 2. Run Development Installer

```bash
# Run the development setup script
./scripts/setup/install_development.sh

# This script will:
# - Install all development dependencies
# - Set up pre-commit hooks
# - Configure development environment
# - Download ML models
# - Initialize databases
# - Run initial tests
```

#### 3. Build from Source

```bash
# Build Swift components
make build-swift

# Build Python components
make build-python

# Run comprehensive tests
make test-all

# Build documentation
make docs
```

### IDE Setup

#### Xcode Configuration

```bash
# Generate Xcode project
swift package generate-xcodeproj

# Open in Xcode
open Vingi.xcodeproj

# Build and run
# Product → Build (⌘B)
# Product → Run (⌘R)
```

#### Visual Studio Code Configuration

```bash
# Install recommended extensions
./scripts/setup/setup_vscode.sh

# Open project
code .

# Use provided launch configurations:
# - Debug Python Components
# - Debug Swift Components
# - Run Tests
```

## Component-Specific Installation

### Core Framework (Swift)

```bash
# Install Swift Package Manager dependencies
swift package update
swift package resolve

# Build framework
swift build -c release

# Run tests
swift test --enable-code-coverage

# Install locally for other projects
swift package install --product VingiCore
```

### ML Backend (Python)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install additional ML dependencies
pip install torch torchvision transformers
pip install sentence-transformers
pip install scikit-learn pandas numpy

# Download pre-trained models
python -m vingi.setup.download_models

# Run ML tests
pytest src/python/tests/ -v
```

### Database Components

#### Neo4j Setup

```bash
# Install Neo4j
brew install neo4j

# Start Neo4j service
brew services start neo4j

# Configure for Vingi
./scripts/setup/configure_neo4j.sh

# Initialize schema
python -m vingi.database.initialize
```

#### SQLite Setup

```bash
# SQLite is pre-installed on macOS
# Initialize application databases
python -m vingi.database.init_sqlite

# Verify database creation
ls -la ~/Library/Application\ Support/Vingi/
```

## Security Configuration

### Encryption Setup

```bash
# Generate encryption keys
vingi security generate-keys

# Configure secure storage
vingi security setup-keychain

# Verify encryption
vingi security test-encryption
```

### Privacy Settings

```bash
# Configure privacy preferences
vingi privacy configure

# Available options:
# - Local-only processing (default)
# - Encrypted cloud sync (optional)
# - Usage analytics (disabled by default)
# - Crash reporting (user choice)
```

### Permissions Setup

```bash
# Grant necessary permissions
vingi permissions request

# Permissions requested:
# - Calendar access (for scheduling optimization)
# - Contacts access (for meeting context)
# - File system access (for file organization)
# - Network access (for research capabilities)
```

## Platform-Specific Installation

### macOS Application

#### System Requirements
- macOS 14.0 (Sonoma) or later
- Apple Silicon (M1/M2) or Intel x86_64
- 8GB RAM minimum, 16GB recommended
- 50GB available storage

#### Installation Steps

```bash
# Download and verify
curl -L -o vingi-macos.dmg https://releases.vingi.dev/v1.0.0/Vingi-macOS.dmg
echo "abc123... vingi-macos.dmg" | shasum -a 256 -c

# Install
hdiutil attach vingi-macos.dmg
sudo cp -R "/Volumes/Vingi/Vingi.app" /Applications/
hdiutil detach "/Volumes/Vingi"

# First launch (will request permissions)
open /Applications/Vingi.app
```

#### Menu Bar Integration

```bash
# Enable menu bar mode
defaults write com.vingi.app MenuBarMode -bool true

# Configure always-on mode
defaults write com.vingi.app AlwaysRunning -bool true
```

### iOS Companion App

#### Installation via App Store

1. Search for "Vingi" in the App Store
2. Tap "Get" to install
3. Open app and follow setup wizard
4. Pair with macOS installation

#### TestFlight Beta (Development)

```bash
# Get TestFlight invitation
curl -X POST https://api.vingi.dev/beta/invite \
  -H "Content-Type: application/json" \
  -d '{"email": "your-email@domain.com"}'

# Install TestFlight and accept invitation
# Download and install Vingi Beta
```

### Command Line Interface

```bash
# Install CLI globally
sudo curl -L -o /usr/local/bin/vingi https://releases.vingi.dev/v1.0.0/vingi-cli
sudo chmod +x /usr/local/bin/vingi

# Verify installation
vingi --version
vingi help

# Set up bash completion
vingi completion bash > /usr/local/etc/bash_completion.d/vingi

# Set up zsh completion
vingi completion zsh > /usr/local/share/zsh/site-functions/_vingi
```

## Configuration and First Run

### Initial Setup Wizard

When launching Vingi for the first time:

```bash
# Run setup wizard
vingi setup

# Or launch GUI setup
open /Applications/Vingi.app
```

**Setup Wizard Steps:**
1. **Welcome and Privacy**: Review privacy policy and data handling
2. **Permissions**: Grant necessary system permissions
3. **Profile Creation**: Set up user profile and preferences
4. **Email Integration**: Configure email accounts (optional)
5. **Calendar Integration**: Connect calendar services
6. **File Management**: Set up file organization preferences
7. **Research Preferences**: Configure research sources and domains
8. **Automation Rules**: Set up initial automation preferences

### Configuration File

Create custom configuration:

```yaml
# ~/.config/vingi/config.yml
user_profile:
  name: "Your Name"
  email: "your-email@domain.com"
  expertise_domains:
    - "machine learning"
    - "software engineering"
    - "research methodology"
  
  privacy_level: "maximum"  # maximum, balanced, minimal
  
automation:
  email_management:
    enabled: true
    auto_response: false
    smart_sorting: true
  
  file_organization:
    enabled: true
    auto_organize_downloads: true
    semantic_naming: true
  
  calendar_optimization:
    enabled: true
    meeting_preparation: true
    smart_scheduling: true

research:
  sources:
    - "arxiv"
    - "google_scholar"
    - "github"
  
  filters:
    min_relevance_score: 0.7
    max_results_per_query: 10
    include_preprints: true

security:
  local_processing_only: true
  encryption_enabled: true
  data_retention_days: 365
```

### Verify Installation

```bash
# Run comprehensive diagnostic
vingi doctor

# Expected output:
# ✓ Core framework installed and running
# ✓ Python ML backend operational
# ✓ Database connections established
# ✓ Security configuration valid
# ✓ All permissions granted
# ✓ Models downloaded and loaded
# ⚠ Email integration not configured
# ℹ Setup completed successfully
```

## Advanced Installation Options

### Custom Build Configuration

For specialized requirements:

```bash
# Build with custom features
./configure \
  --enable-gpu-acceleration \
  --disable-cloud-sync \
  --with-custom-models=/path/to/models \
  --prefix=/opt/vingi

make install
```

### Docker Deployment

For containerized deployment:

```bash
# Pull official image
docker pull vingi/vingi:latest

# Run with volume mounts for data persistence
docker run -d \
  --name vingi \
  -v vingi_data:/app/data \
  -v ~/.vingi:/app/config \
  -p 8080:8080 \
  vingi/vingi:latest

# Access web interface
open http://localhost:8080
```

### Server Deployment

For multi-user server deployment:

```bash
# Install server edition
curl -fsSL https://install.vingi.dev/server | bash

# Configure multi-user settings
sudo vingi-server configure --multi-user

# Start server daemon
sudo systemctl enable vingi-server
sudo systemctl start vingi-server
```

## Troubleshooting Installation

### Common Issues

#### Permission Denied Errors

```bash
# Fix file permissions
sudo chown -R $(whoami) /Applications/Vingi.app
chmod +x /Applications/Vingi.app/Contents/MacOS/Vingi

# Fix CLI permissions
sudo chown $(whoami) /usr/local/bin/vingi
chmod +x /usr/local/bin/vingi
```

#### Dependency Conflicts

```bash
# Reset Python environment
pipenv --rm
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Reset Swift package cache
swift package reset
swift package update
```

#### Database Connection Issues

```bash
# Restart database services
brew services restart neo4j
brew services restart redis

# Reinitialize databases
./scripts/setup/reset_databases.sh
```

#### Model Download Failures

```bash
# Clear model cache
rm -rf data/models/
mkdir -p data/models/{embeddings,classifiers,custom}

# Re-download models
python -m vingi.setup.download_models --force

# Verify model integrity
python -m vingi.setup.verify_models
```

### Getting Help

#### Built-in Diagnostics

```bash
# Run diagnostic suite
vingi doctor --verbose

# Test specific components
vingi test --component=ml
vingi test --component=database
vingi test --component=security
```

#### Log Analysis

```bash
# View application logs
vingi logs --tail --level=error

# View system logs
log show --predicate 'subsystem == "com.vingi.app"' --last 1h

# Export logs for support
vingi logs --export --output=vingi-logs.zip
```

#### Community Support

- **Documentation**: [https://docs.vingi.dev](https://docs.vingi.dev)
- **Issue Tracker**: [https://github.com/yourusername/vingi/issues](https://github.com/yourusername/vingi/issues)
- **Discussions**: [https://github.com/yourusername/vingi/discussions](https://github.com/yourusername/vingi/discussions)
- **Email Support**: support@vingi.dev

## Uninstallation

### Complete Removal

```bash
# Stop running services
vingi stop

# Remove application
sudo rm -rf /Applications/Vingi.app

# Remove CLI tools
sudo rm /usr/local/bin/vingi

# Remove user data (optional - this deletes all your data!)
rm -rf ~/Library/Application\ Support/Vingi/
rm -rf ~/.config/vingi/

# Remove database (optional)
brew services stop neo4j
rm -rf /opt/homebrew/var/neo4j/
```

### Partial Removal (Keep Data)

```bash
# Remove application but keep user data
sudo rm -rf /Applications/Vingi.app
sudo rm /usr/local/bin/vingi

# Data remains in:
# - ~/Library/Application Support/Vingi/
# - ~/.config/vingi/
```

This installation guide ensures a smooth, secure, and reliable setup process for all Vingi users, from casual users to developers and system administrators.
