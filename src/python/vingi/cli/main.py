"""
Vingi CLI - Command Line Interface for Personal AI Assistant

This module provides the main CLI interface for Vingi.
"""

import click
import sys
import asyncio
from pathlib import Path
from typing import Optional
import json
import yaml

from .. import __version__, get_data_dir, get_config_dir
from ..utils.logging import get_logger, configure_logging


logger = get_logger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option(
    '--config', 
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file'
)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', is_flag=True, help='Enable debug output')
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool, debug: bool):
    """Vingi Personal AI Assistant CLI
    
    A privacy-first AI assistant for automating personal affairs 
    and reducing cognitive load.
    """
    # Configure logging based on options
    if debug:
        log_level = 'DEBUG'
    elif verbose:
        log_level = 'INFO'
    else:
        log_level = 'WARNING'
    
    configure_logging(level=log_level, enable_console=True)
    
    # Store configuration in context
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    ctx.obj['debug'] = debug
    
    logger.info(f"Vingi CLI v{__version__} starting")


@cli.command()
@click.option('--force', is_flag=True, help='Force re-initialization')
@click.option('--user-name', prompt='Your name', help='Your full name')
@click.option('--email', help='Your email address (optional)')
@click.option(
    '--privacy-level',
    type=click.Choice(['minimal', 'balanced', 'maximum']),
    default='maximum',
    help='Privacy level (default: maximum)'
)
@click.pass_context
def init(ctx: click.Context, force: bool, user_name: str, email: Optional[str], privacy_level: str):
    """Initialize Vingi configuration and data directories."""
    
    config_dir = get_config_dir()
    data_dir = get_data_dir()
    config_file = config_dir / "config.yml"
    
    if config_file.exists() and not force:
        click.echo("Vingi is already initialized. Use --force to re-initialize.")
        return
    
    click.echo(f"Initializing Vingi in {config_dir}")
    
    # Create basic configuration
    config = {
        'user_profile': {
            'name': user_name,
            'email': email,
            'expertise_domains': [],
            'privacy_level': privacy_level
        },
        'privacy_settings': {
            'privacy_level': privacy_level,
            'local_processing_only': privacy_level == 'maximum',
            'data_retention_days': 365,
            'analytics_enabled': False,
            'encryption_required': True
        },
        'automation_preferences': {
            'email_management': {'enabled': True, 'smart_sorting': True},
            'file_organization': {'enabled': True, 'auto_organize_downloads': True},
            'calendar_optimization': {'enabled': True, 'meeting_preparation': True}
        },
        'data_paths': {
            'data_dir': str(data_dir),
            'models_dir': str(data_dir / 'models'),
            'cache_dir': str(data_dir / 'cache'),
            'logs_dir': str(data_dir / 'logs')
        }
    }
    
    # Write configuration
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Create data directories
    (data_dir / 'models').mkdir(parents=True, exist_ok=True)
    (data_dir / 'cache').mkdir(parents=True, exist_ok=True)
    (data_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (data_dir / 'context').mkdir(parents=True, exist_ok=True)
    
    click.echo(f"✓ Configuration created: {config_file}")
    click.echo(f"✓ Data directory: {data_dir}")
    click.echo("✓ Vingi initialization complete!")
    
    logger.info(f"Vingi initialized for user: {user_name}")


@cli.command()
@click.option(
    '--component',
    type=click.Choice(['all', 'core', 'ml', 'database', 'security']),
    default='all',
    help='Component to test'
)
@click.pass_context
def doctor(ctx: click.Context, component: str):
    """Run system diagnostic checks."""
    
    click.echo("🔍 Running Vingi system diagnostic...")
    
    issues = []
    checks_passed = 0
    total_checks = 0
    
    def check(name: str, condition: bool, fix_suggestion: str = ""):
        nonlocal checks_passed, total_checks
        total_checks += 1
        if condition:
            click.echo(f"✓ {name}")
            checks_passed += 1
        else:
            click.echo(f"✗ {name}")
            if fix_suggestion:
                issues.append(f"  Fix: {fix_suggestion}")
    
    # Basic system checks
    if component in ['all', 'core']:
        click.echo("\n📋 Core System:")
        
        config_dir = get_config_dir()
        data_dir = get_data_dir()
        config_file = config_dir / "config.yml"
        
        check(
            "Configuration directory exists",
            config_dir.exists(),
            f"Run 'vingi init' to initialize"
        )
        
        check(
            "Configuration file exists",
            config_file.exists(),
            f"Run 'vingi init' to create configuration"
        )
        
        check(
            "Data directory exists",
            data_dir.exists(),
            f"Data directory will be created automatically"
        )
        
        check(
            "Models directory exists",
            (data_dir / 'models').exists(),
            f"Run 'vingi init' to create directory structure"
        )
        
        check(
            "Cache directory exists",
            (data_dir / 'cache').exists(),
            f"Run 'vingi init' to create directory structure"
        )
    
    # Python environment checks
    if component in ['all', 'ml']:
        click.echo("\n🐍 Python Environment:")
        
        try:
            import torch
            check("PyTorch available", True)
        except ImportError:
            check("PyTorch available", False, "pip install torch")
        
        try:
            import transformers
            check("Transformers available", True)
        except ImportError:
            check("Transformers available", False, "pip install transformers")
        
        try:
            import sentence_transformers
            check("Sentence Transformers available", True)
        except ImportError:
            check("Sentence Transformers available", False, "pip install sentence-transformers")
        
        try:
            import sklearn
            check("Scikit-learn available", True)
        except ImportError:
            check("Scikit-learn available", False, "pip install scikit-learn")
    
    # Database checks
    if component in ['all', 'database']:
        click.echo("\n🗄️  Database:")
        
        try:
            import neo4j
            check("Neo4j driver available", True)
        except ImportError:
            check("Neo4j driver available", False, "pip install neo4j")
        
        import sqlite3
        check("SQLite available", True)
    
    # Security checks
    if component in ['all', 'security']:
        click.echo("\n🔒 Security:")
        
        try:
            import cryptography
            check("Cryptography library available", True)
        except ImportError:
            check("Cryptography library available", False, "pip install cryptography")
        
        try:
            import keyring
            check("Keyring library available", True)
        except ImportError:
            check("Keyring library available", False, "pip install keyring")
    
    # Summary
    click.echo(f"\n📊 Diagnostic Summary:")
    click.echo(f"   Checks passed: {checks_passed}/{total_checks}")
    
    if issues:
        click.echo(f"\n⚠️  Issues found:")
        for issue in issues:
            click.echo(issue)
        sys.exit(1)
    else:
        click.echo("✅ All checks passed!")


@cli.command()
@click.option('--tail', '-f', is_flag=True, help='Follow log output')
@click.option('--lines', '-n', default=50, help='Number of lines to show')
@click.option(
    '--level',
    type=click.Choice(['debug', 'info', 'warning', 'error']),
    help='Filter by log level'
)
@click.pass_context
def logs(ctx: click.Context, tail: bool, lines: int, level: Optional[str]):
    """View Vingi application logs."""
    
    data_dir = get_data_dir()
    log_file = data_dir / 'logs' / 'vingi.log'
    
    if not log_file.exists():
        click.echo("No log file found. Start Vingi to generate logs.")
        return
    
    if tail:
        # Follow log file
        click.echo(f"Following {log_file} (Ctrl+C to stop)")
        try:
            import subprocess
            cmd = ['tail', '-f', str(log_file)]
            if level:
                cmd = ['tail', '-f', str(log_file)] + ['|', 'grep', '-i', level]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            click.echo("\nLog following stopped.")
    else:
        # Show last N lines
        try:
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
                
                if level:
                    log_lines = [line for line in log_lines if level.upper() in line]
                
                # Show last N lines
                for line in log_lines[-lines:]:
                    click.echo(line.rstrip())
                    
        except Exception as e:
            click.echo(f"Error reading log file: {e}")


@cli.command()
@click.argument('query')
@click.option('--context', help='Additional context for the query')
@click.option('--json-output', is_flag=True, help='Output response as JSON')
@click.pass_context
def query(ctx: click.Context, query: str, context: Optional[str], json_output: bool):
    """Process a query using Vingi's intelligence engine."""
    
    # This would integrate with the actual intelligence engine
    # For now, just echo back the query
    
    click.echo(f"Processing query: {query}")
    
    if context:
        click.echo(f"Context: {context}")
    
    # Mock response for now
    response = {
        'query': query,
        'response': f"This is a mock response to: {query}",
        'confidence': 0.8,
        'timestamp': '2024-01-01T00:00:00Z',
        'suggestions': [
            {'action': 'create_automation', 'description': 'Create automation for similar queries'}
        ]
    }
    
    if json_output:
        click.echo(json.dumps(response, indent=2))
    else:
        click.echo(f"\nResponse: {response['response']}")
        click.echo(f"Confidence: {response['confidence']:.1%}")
        
        if response['suggestions']:
            click.echo("\nSuggestions:")
            for suggestion in response['suggestions']:
                click.echo(f"  • {suggestion['description']}")


@cli.command()
@click.option('--export', is_flag=True, help='Export configuration')
@click.option('--import-file', type=click.Path(exists=True), help='Import configuration from file')
@click.pass_context
def config(ctx: click.Context, export: bool, import_file: Optional[Path]):
    """Manage Vingi configuration."""
    
    config_file = get_config_dir() / "config.yml"
    
    if export:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = f.read()
            click.echo(config_data)
        else:
            click.echo("No configuration file found. Run 'vingi init' first.")
    
    elif import_file:
        click.echo(f"Importing configuration from {import_file}")
        # Here you would validate and import the configuration
        click.echo("Configuration import not yet implemented.")
    
    else:
        # Show current configuration
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            click.echo("Current configuration:")
            click.echo(yaml.dump(config_data, default_flow_style=False, indent=2))
        else:
            click.echo("No configuration file found. Run 'vingi init' first.")


@cli.command()
@click.pass_context
def status(ctx: click.Context):
    """Show Vingi system status."""
    
    click.echo("📊 Vingi System Status")
    click.echo("=" * 40)
    
    # Version info
    click.echo(f"Version: {__version__}")
    
    # Directory info
    config_dir = get_config_dir()
    data_dir = get_data_dir()
    
    click.echo(f"Config directory: {config_dir}")
    click.echo(f"Data directory: {data_dir}")
    
    # Configuration status
    config_file = config_dir / "config.yml"
    if config_file.exists():
        click.echo("✓ Configuration: Found")
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            user_name = config.get('user_profile', {}).get('name', 'Unknown')
            privacy_level = config.get('privacy_settings', {}).get('privacy_level', 'Unknown')
            click.echo(f"  User: {user_name}")
            click.echo(f"  Privacy level: {privacy_level}")
        except Exception as e:
            click.echo(f"  Warning: Could not read configuration: {e}")
    else:
        click.echo("✗ Configuration: Not found (run 'vingi init')")
    
    # Data directory status
    if data_dir.exists():
        click.echo("✓ Data directory: Found")
        
        models_dir = data_dir / 'models'
        cache_dir = data_dir / 'cache'
        logs_dir = data_dir / 'logs'
        
        click.echo(f"  Models: {len(list(models_dir.glob('*'))) if models_dir.exists() else 0} items")
        click.echo(f"  Cache: {len(list(cache_dir.glob('*'))) if cache_dir.exists() else 0} items")
        
        if logs_dir.exists():
            log_files = list(logs_dir.glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                click.echo(f"  Latest log: {latest_log.name}")
    else:
        click.echo("✗ Data directory: Not found")


if __name__ == '__main__':
    cli() 