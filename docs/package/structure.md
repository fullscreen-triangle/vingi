# Vingi Project Structure

## Overview

The Vingi project follows a modular, multi-platform architecture designed for scalability, maintainability, and privacy-first operation. The structure supports both macOS and iOS implementations while maintaining clear separation between concerns.

## Root Directory Structure

```
vingi/
├── README.md                          # Main project documentation
├── LICENSE                            # MIT license file
├── Package.swift                      # Swift package manifest
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Python project configuration
├── Makefile                          # Build automation
├── .gitignore                        # Git ignore patterns
├── .swiftlint.yml                    # Swift linting configuration
├── .pre-commit-config.yaml           # Pre-commit hooks
│
├── docs/                             # Documentation directory
│   ├── package/                      # Package-specific docs
│   │   ├── structure.md              # This file
│   │   ├── setup.md                  # Setup instructions
│   │   ├── installation.md           # Installation guide
│   │   └── components.md             # Component specifications
│   ├── assets/                       # Documentation assets
│   │   ├── vingi_logo.png            # Project logo
│   │   ├── architecture_diagram.svg  # System architecture
│   │   └── flow_charts/              # Process flow diagrams
│   ├── api/                          # API documentation
│   │   ├── core_api.md               # Core API reference
│   │   ├── ios_api.md                # iOS-specific APIs
│   │   └── macos_api.md              # macOS-specific APIs
│   └── research/                     # Research and theoretical docs
│       ├── cognitive_load_theory.md  # Theoretical foundation
│       ├── information_theory.md     # Mathematical models
│       └── citations.bib             # Bibliography
│
├── config/                           # Configuration files
│   ├── config.template.yml           # Configuration template
│   ├── development.yml               # Development settings
│   ├── production.yml                # Production settings
│   ├── logging.yml                   # Logging configuration
│   └── security/                     # Security configurations
│       ├── encryption.yml            # Encryption settings
│       └── privacy.yml               # Privacy policy settings
│
├── src/                              # Source code directory
│   ├── core/                         # Core framework (Swift)
│   │   ├── VingiCore/                # Main core module
│   │   │   ├── Sources/              # Swift source files
│   │   │   │   ├── VingiCore/        # Core functionality
│   │   │   │   │   ├── VingiCore.swift
│   │   │   │   │   ├── Context/      # Context management
│   │   │   │   │   ├── Intelligence/ # AI/ML components
│   │   │   │   │   ├── Automation/   # Task automation
│   │   │   │   │   ├── Security/     # Security & encryption
│   │   │   │   │   └── Utils/        # Utility functions
│   │   │   └── Tests/                # Unit tests
│   │   │       └── VingiCoreTests/
│   │   ├── VingiMacOS/               # macOS-specific module
│   │   └── VingiIOS/                 # iOS-specific module
│   │
│   ├── python/                       # Python ML/AI components
│   │   ├── vingi/                    # Main Python package
│   │   │   ├── __init__.py
│   │   │   ├── core/                 # Core ML algorithms
│   │   │   │   ├── __init__.py
│   │   │   │   ├── pattern_recognition.py
│   │   │   │   ├── relevance_scoring.py
│   │   │   │   ├── temporal_analysis.py
│   │   │   │   └── context_graph.py
│   │   │   ├── nlp/                  # Natural language processing
│   │   │   │   ├── __init__.py
│   │   │   │   ├── email_classifier.py
│   │   │   │   ├── text_summarizer.py
│   │   │   │   └── intent_recognition.py
│   │   │   ├── research/             # Research automation
│   │   │   │   ├── __init__.py
│   │   │   │   ├── search_agent.py
│   │   │   │   ├── citation_parser.py
│   │   │   │   └── knowledge_synthesis.py
│   │   │   ├── automation/           # Task automation
│   │   │   │   ├── __init__.py
│   │   │   │   ├── file_manager.py
│   │   │   │   ├── email_processor.py
│   │   │   │   └── calendar_optimizer.py
│   │   │   └── utils/                # Utility functions
│   │   │       ├── __init__.py
│   │   │       ├── encryption.py
│   │   │       ├── logging.py
│   │   │       └── config_loader.py
│   │   └── tests/                    # Python tests
│   │       ├── test_core/
│   │       ├── test_nlp/
│   │       ├── test_research/
│   │       └── test_automation/
│   │
│   └── bridge/                       # Python-Swift bridge
│       ├── VingiBridge/              # Swift bridge module
│       │   ├── Sources/
│       │   │   └── VingiBridge/
│       │   │       ├── PythonBridge.swift
│       │   │       ├── MLModelManager.swift
│       │   │       └── DataTransfer.swift
│       │   └── Tests/
│       └── python_interface/         # Python interface
│           ├── __init__.py
│           ├── swift_bridge.py
│           └── model_server.py
│
├── apps/                             # Application implementations
│   ├── macos/                        # macOS application
│   │   ├── Vingi.xcodeproj/          # Xcode project
│   │   ├── Vingi/                    # App source
│   │   │   ├── AppDelegate.swift
│   │   │   ├── SceneDelegate.swift
│   │   │   ├── Views/                # SwiftUI views
│   │   │   │   ├── MainView.swift
│   │   │   │   ├── ContextView.swift
│   │   │   │   ├── AutomationView.swift
│   │   │   │   └── SettingsView.swift
│   │   │   ├── Controllers/          # View controllers
│   │   │   ├── Models/               # Data models
│   │   │   └── Resources/            # App resources
│   │   │       ├── Assets.xcassets
│   │   │       ├── Info.plist
│   │   │       └── Localizable.strings
│   │   └── VingiTests/               # App tests
│   │
│   ├── ios/                          # iOS application
│   │   ├── Vingi.xcodeproj/
│   │   ├── Vingi/
│   │   │   ├── AppDelegate.swift
│   │   │   ├── SceneDelegate.swift
│   │   │   ├── Views/
│   │   │   ├── Controllers/
│   │   │   ├── Models/
│   │   │   └── Resources/
│   │   └── VingiTests/
│   │
│   └── cli/                          # Command-line interface
│       ├── vingi_cli/                # CLI implementation
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── commands/             # CLI commands
│       │   │   ├── __init__.py
│       │   │   ├── init.py           # Project initialization
│       │   │   ├── research.py       # Research commands
│       │   │   ├── automate.py       # Automation commands
│       │   │   └── status.py         # Status commands
│       │   └── utils/
│       └── setup.py                  # CLI package setup
│
├── data/                             # Data directory (local storage)
│   ├── models/                       # ML models storage
│   │   ├── embeddings/               # Embedding models
│   │   ├── classifiers/              # Classification models
│   │   └── custom/                   # User-trained models
│   ├── cache/                        # Application cache
│   │   ├── search_results/
│   │   ├── processed_emails/
│   │   └── file_metadata/
│   ├── context/                      # Context storage
│   │   ├── knowledge_graph.db        # Neo4j database
│   │   ├── temporal_index.db         # Time-based index
│   │   └── user_preferences.json     # User settings
│   └── logs/                         # Application logs
│       ├── application.log
│       ├── security.log
│       └── performance.log
│
├── scripts/                          # Utility scripts
│   ├── setup/                        # Setup scripts
│   │   ├── setup_database.sh         # Database initialization
│   │   ├── install_dependencies.sh   # Dependency installation
│   │   └── configure_environment.sh  # Environment setup
│   ├── development/                  # Development tools
│   │   ├── run_tests.sh              # Test runner
│   │   ├── lint_code.sh              # Code linting
│   │   └── generate_docs.sh          # Documentation generation
│   ├── deployment/                   # Deployment scripts
│   │   ├── build_release.sh          # Release build
│   │   ├── create_installer.sh       # Installer creation
│   │   └── update_version.sh         # Version management
│   └── maintenance/                  # Maintenance scripts
│       ├── cleanup_cache.sh          # Cache cleanup
│       ├── backup_data.sh            # Data backup
│       └── rotate_logs.sh            # Log rotation
│
├── tests/                            # Integration tests
│   ├── integration/                  # Cross-component tests
│   │   ├── test_swift_python_bridge.py
│   │   ├── test_end_to_end_workflows.py
│   │   └── test_cross_device_sync.py
│   ├── performance/                  # Performance tests
│   │   ├── test_latency.py
│   │   ├── test_memory_usage.py
│   │   └── test_battery_impact.py
│   ├── security/                     # Security tests
│   │   ├── test_encryption.py
│   │   ├── test_privacy_compliance.py
│   │   └── test_access_controls.py
│   └── fixtures/                     # Test data
│       ├── sample_emails/
│       ├── test_documents/
│       └── mock_responses/
│
├── tools/                            # Development tools
│   ├── code_generation/              # Code generators
│   │   ├── generate_api_bindings.py
│   │   └── generate_model_interfaces.py
│   ├── analysis/                     # Code analysis tools
│   │   ├── complexity_analyzer.py
│   │   └── dependency_graph.py
│   └── monitoring/                   # Monitoring tools
│       ├── performance_monitor.py
│       └── resource_tracker.py
│
└── vendor/                           # Third-party dependencies
    ├── swift_packages/               # Swift package cache
    ├── python_wheels/                # Python package cache
    └── models/                       # Pre-trained model files
        ├── sentence_transformers/
        ├── classification_models/
        └── language_models/
```

## Directory Descriptions

### Core Directories

#### `/src/core/`
Contains the main Swift framework that provides core functionality across all platforms. Organized into modular components for maintainability and testing.

#### `/src/python/`
Houses all Python-based machine learning and natural language processing components. Designed for high-performance data processing and model inference.

#### `/src/bridge/`
Implements the critical Python-Swift bridge that enables seamless integration between the ML backend and native UI frontend.

#### `/apps/`
Platform-specific application implementations that leverage the core framework while providing native user experiences.

### Configuration Management

#### `/config/`
Centralized configuration management with environment-specific settings and security configurations. Supports hot-reloading for development.

#### `/data/`
Local data storage following privacy-first principles. All data remains on-device with strong encryption.

### Development Infrastructure

#### `/scripts/`
Comprehensive automation scripts for development, testing, deployment, and maintenance workflows.

#### `/tests/`
Multi-layered testing strategy including unit tests, integration tests, performance tests, and security validation.

#### `/tools/`
Development productivity tools for code generation, analysis, and monitoring.

## File Naming Conventions

### Swift Files
- **Classes**: `PascalCase` (e.g., `ContextManager.swift`)
- **Protocols**: `PascalCase` with descriptive suffix (e.g., `DataSourceProtocol.swift`)
- **Extensions**: `BaseClass+Extension.swift` (e.g., `String+Encryption.swift`)
- **Views**: `PascalCase` with `View` suffix (e.g., `ContextView.swift`)

### Python Files
- **Modules**: `snake_case` (e.g., `pattern_recognition.py`)
- **Classes**: `PascalCase` within files (e.g., `class EmailClassifier`)
- **Functions**: `snake_case` (e.g., `def analyze_temporal_patterns()`)
- **Tests**: `test_` prefix (e.g., `test_relevance_scoring.py`)

### Configuration Files
- **YAML**: `.yml` extension for better readability
- **Environment-specific**: Environment prefix (e.g., `development.yml`)
- **Templates**: `.template` suffix (e.g., `config.template.yml`)

## Module Dependencies

### Core Dependencies
```
VingiCore (Swift)
├── Foundation
├── Combine
├── CryptoKit
├── OSLog
└── Network

VingiPython
├── torch >= 2.0.0
├── transformers >= 4.21.0
├── scikit-learn >= 1.1.0
├── numpy >= 1.23.0
├── pandas >= 1.5.0
├── neo4j >= 5.0.0
└── cryptography >= 37.0.0
```

### Platform-Specific Dependencies

#### macOS
```
VingiMacOS
├── AppKit
├── EventKit
├── ContactsUI
├── Quartz
└── ApplicationServices
```

#### iOS
```
VingiIOS
├── UIKit
├── SwiftUI
├── EventKit
├── Contacts
└── BackgroundTasks
```

## Data Flow Architecture

### Information Processing Pipeline
```
User Input → Context Analysis → Pattern Recognition → Decision Engine → Action Execution → Feedback Loop
     ↑                                     ↓
External APIs ←→ Relevance Scoring ←→ Historical Data
```

### Cross-Platform Synchronization
```
iOS Device ←→ Local Network ←→ macOS Device
     ↓              ↓              ↓
Encrypted State   Vector Clock   Encrypted State
   Storage      Synchronization     Storage
```

## Security Architecture

### Data Protection Layers
1. **Application Layer**: Input validation, secure coding practices
2. **Data Layer**: AES-256 encryption, secure key management
3. **Network Layer**: TLS 1.3, certificate pinning
4. **System Layer**: Sandboxing, hardware-backed encryption

### Access Control Matrix
| Component | Local Access | Network Access | External API | User Data |
|-----------|-------------|----------------|--------------|-----------|
| Core Framework | Full | Restricted | None | Encrypted |
| Python ML | Sandboxed | None | Academic APIs | Processed |
| UI Applications | Standard | Local Network | Apple APIs | Display Only |
| CLI Tools | Limited | None | None | Encrypted |

## Performance Considerations

### Memory Management
- **Swift**: Automatic Reference Counting (ARC)
- **Python**: Garbage collection with memory limits
- **Bridge**: Efficient data serialization with minimal copying

### Processing Distribution
- **Real-time**: Swift components (< 100ms latency)
- **Background**: Python ML processing (2-30s acceptable)
- **Batch**: Overnight processing for heavy analysis

### Storage Optimization
- **Hot Data**: In-memory cache (< 1GB)
- **Warm Data**: SQLite databases (< 5GB)
- **Cold Data**: Compressed archives (unlimited, user-managed)

## Extensibility Framework

### Plugin Architecture
```
Core Framework
├── Protocol Definitions
├── Plugin Registry
├── Sandbox Environment
└── API Gateway

Third-Party Plugins
├── Email Providers
├── Calendar Systems
├── File Sync Services
└── Research Databases
```

### Custom Model Integration
- **Training Pipeline**: Local model training with user data
- **Model Registry**: Versioned model storage and rollback
- **A/B Testing**: Performance comparison framework
- **Hot Swapping**: Runtime model replacement

This structure ensures scalability, maintainability, and privacy while supporting the sophisticated AI capabilities that make Vingi an effective personal cognitive load optimization system.
