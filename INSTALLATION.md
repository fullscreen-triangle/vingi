# Vingi Installation Guide

This guide will help you install and set up the Vingi Personal Cognitive Load Optimization Framework.

## Prerequisites

- **Python 3.11 or higher**
- **macOS 14.0+** (primary platform)
- **8GB+ RAM** recommended
- **1GB free disk space**

## Installation Methods

### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/vingi-ai/vingi.git
cd vingi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install Vingi in development mode
pip install -e .
```

### Option 2: Install Core Dependencies Only

For a minimal installation with just the core features:

```bash
pip install numpy click PyYAML python-dateutil
```

## Configuration

### First Run Setup

1. **Initialize Vingi**:
   ```bash
   vingi status
   ```
   This creates the default configuration and data directories.

2. **View Configuration**:
   ```bash
   vingi config show
   ```

3. **Customize Settings** (optional):
   ```bash
   # Adjust pattern detection sensitivity
   vingi config set-patterns --threshold 0.8

   # Customize relevance scoring weights
   vingi config set-weights --personal 0.4 --temporal 0.3

   # Set intervention style
   vingi config set-global --style moderate
   ```

## Quick Start

### 1. Record Your First Behavior

Track a decision-making process:
```bash
vingi patterns add-behavior \
  --action research \
  --domain transportation \
  --duration 30 \
  --complexity medium \
  --metadata '{"task_id": "train_booking"}'
```

### 2. Add Preferences

Tell Vingi what you like and dislike:
```bash
# Things you like
vingi context add-preference --item "fresh bread" --type like --strength 0.9 --domain food
vingi context add-preference --item "fast trains" --type like --strength 0.8 --domain transportation

# Things you dislike
vingi context add-preference --item "processed food" --type dislike --strength 0.7 --domain food
```

### 3. Track Temporal Patterns

Record work sessions to build your productivity profile:
```bash
vingi temporal add-event \
  --event-type coding \
  --duration 120 \
  --energy 0.8 \
  --focus 0.9 \
  --complexity high \
  --completed true \
  --interruptions 1
```

### 4. Get Recommendations

Once you have some data, get personalized recommendations:
```bash
# Check for cognitive patterns
vingi patterns status
vingi patterns interventions

# Find optimal work times
vingi temporal optimal-times --task-type coding --duration 120

# Check energy predictions
vingi temporal predict-energy

# View your preferences
vingi context show-preferences
```

## Data Storage

Vingi stores all data locally for privacy:

- **Configuration**: `~/Library/Application Support/Vingi/config.yaml`
- **Context Graph**: `~/Library/Application Support/Vingi/context.db`
- **Temporal Data**: `~/Library/Application Support/Vingi/temporal_data.json`

## Configuration Profiles

You can create multiple configuration profiles for different use cases:

```bash
# Save current config as "work" profile
vingi config save-profile work

# Create a more sensitive analysis profile
vingi config set-patterns --threshold 0.6
vingi config save-profile sensitive

# Switch between profiles
vingi config load-profile work
vingi config list-profiles
```

## CLI Command Reference

### Core Commands

- `vingi status` - Show overall system status
- `vingi config show` - Display current configuration
- `vingi export-data` - Export all your data

### Pattern Detection

- `vingi patterns add-behavior` - Record behavior for analysis
- `vingi patterns status` - View detected patterns
- `vingi patterns interventions` - Get recommendations

### Context Management

- `vingi context add-preference` - Record preferences
- `vingi context show-preferences` - View preferences
- `vingi context analyze` - Analyze context patterns

### Temporal Analysis

- `vingi temporal add-event` - Record time-based events
- `vingi temporal optimal-times` - Find optimal scheduling
- `vingi temporal predict-energy` - Energy level predictions
- `vingi temporal analyze-productivity` - Productivity analysis

### Information Relevance

- `vingi relevance score` - Score information relevance

### Configuration

- `vingi config set-patterns` - Adjust pattern detection
- `vingi config set-weights` - Customize relevance weights
- `vingi config set-temporal` - Temporal analysis settings
- `vingi config validate` - Check configuration validity

## Integration with Development Environment

### PyCharm Integration

1. Set up the Vingi project in PyCharm
2. Configure the Python interpreter to use your virtual environment
3. Run individual modules or the demo script
4. Use PyCharm's debugger to step through pattern detection

### VS Code Integration

1. Open the Vingi folder in VS Code
2. Install the Python extension
3. Select the Vingi virtual environment as your interpreter
4. Use the integrated terminal to run CLI commands

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure you're in the right directory and venv is activated
   which python  # Should show venv path
   pip list | grep vingi
   ```

2. **Permission Errors**:
   ```bash
   # Check data directory permissions
   ls -la ~/Library/Application\ Support/Vingi/
   ```

3. **Configuration Issues**:
   ```bash
   # Reset to defaults if needed
   vingi config reset
   vingi config validate
   ```

### Getting Help

- Use `vingi --help` for command overview
- Use `vingi [command] --help` for specific command help
- Check the README.md for detailed framework documentation
- Review the demo script in `examples/cognitive_pattern_demo.py`

## Next Steps

1. **Run the Demo**: `python examples/cognitive_pattern_demo.py`
2. **Start Tracking**: Begin recording your daily behaviors and preferences
3. **Analyze Patterns**: Review weekly pattern reports and interventions
4. **Customize**: Adjust settings based on your personal needs
5. **Integrate**: Connect with your calendar, email, and other productivity tools

## Privacy and Security

- All data is stored locally on your machine
- No data is transmitted to external servers
- Configuration and data files are stored in your user directory
- You have full control over your data export and backup

For more detailed information, see the main README.md file. 