#!/usr/bin/env python3
"""
Vingi Command Line Interface

Provides command-line access to all Vingi cognitive optimization features.
"""

import click
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

from vingi.core import (
    TemporalPatternRecognizer,
    ContextGraphManager,
    RelevanceScorer,
    TemporalAnalyzer,
    BehaviorData,
    CognitivePatternType,
    InformationItem,
    TemporalEvent,
    TaskComplexity,
    NodeType
)

from vingi.config import ConfigManager, get_config_manager


class VingiCLI:
    """Main CLI class for Vingi framework."""
    
    def __init__(self):
        """Initialize CLI with core components."""
        self.config_manager = get_config_manager()
        config = self.config_manager.config
        
        self.data_dir = Path(config.data_directory).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components with configuration
        self.pattern_recognizer = TemporalPatternRecognizer(
            window_size=config.pattern_detection.window_size,
            confidence_threshold=config.pattern_detection.confidence_threshold
        )
        self.context_manager = ContextGraphManager()
        self.temporal_analyzer = TemporalAnalyzer()
        self.relevance_scorer = RelevanceScorer(self.context_manager, self.pattern_recognizer)
    
    def save_all(self):
        """Save all component data."""
        self.context_manager.save()
        self.temporal_analyzer.save_data()


@click.group()
@click.pass_context
def cli(ctx):
    """Vingi Personal Cognitive Load Optimization Framework"""
    ctx.ensure_object(dict)
    ctx.obj['vingi'] = VingiCLI()


@cli.group()
def patterns():
    """Cognitive pattern detection and analysis."""
    pass


@patterns.command()
@click.option('--action', required=True, help='Action type (research, planning, choice, etc.)')
@click.option('--domain', default='general', help='Domain (food, transportation, etc.)')
@click.option('--duration', type=int, help='Duration in minutes')
@click.option('--complexity', type=click.Choice(['low', 'medium', 'high']), default='medium')
@click.option('--metadata', help='Additional metadata as JSON string')
@click.pass_context
def add_behavior(ctx, action, domain, duration, complexity, metadata):
    """Add a behavior event for pattern analysis."""
    vingi = ctx.obj['vingi']
    
    # Parse metadata
    meta = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            click.echo("Invalid JSON in metadata, ignoring.")
    
    # Create behavior data
    behavior = BehaviorData(
        timestamp=datetime.now(),
        action_type=action,
        duration=timedelta(minutes=duration) if duration else None,
        task_complexity=complexity,
        domain=domain,
        metadata=meta
    )
    
    # Add to recognizer
    detected_pattern = vingi.pattern_recognizer.add_behavior_event(behavior)
    
    if detected_pattern:
        click.echo(f"🧠 Pattern detected: {detected_pattern.pattern_type.value}")
        click.echo(f"   Confidence: {detected_pattern.confidence:.2f}")
        click.echo(f"   Severity: {detected_pattern.severity}")
        click.echo("   Suggested interventions:")
        for intervention in detected_pattern.intervention_suggestions:
            click.echo(f"   • {intervention}")
    else:
        click.echo("✅ Behavior recorded, no patterns detected.")


@patterns.command()
@click.pass_context
def status(ctx):
    """Show current pattern detection status."""
    vingi = ctx.obj['vingi']
    
    summary = vingi.pattern_recognizer.get_pattern_summary()
    
    click.echo("🧠 Cognitive Pattern Status")
    click.echo("=" * 30)
    click.echo(f"Total patterns detected: {summary['total_patterns_detected']}")
    click.echo(f"Behavior events analyzed: {summary['behavior_events_analyzed']}")
    click.echo(f"Recent patterns (last 7 days): {summary['recent_patterns']}")
    
    if summary['pattern_type_distribution']:
        click.echo("\nPattern distribution:")
        for pattern_type, count in summary['pattern_type_distribution'].items():
            click.echo(f"  {pattern_type}: {count}")
    
    click.echo(f"\nAverage confidence: {summary.get('average_confidence', 0):.2f}")


@patterns.command()
@click.pass_context
def interventions(ctx):
    """Get current intervention recommendations."""
    vingi = ctx.obj['vingi']
    
    recommendations = vingi.pattern_recognizer.get_intervention_recommendations()
    
    click.echo("💡 Current Intervention Recommendations")
    click.echo("=" * 40)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            click.echo(f"{i}. {rec}")
    else:
        click.echo("No current recommendations. Keep up the good work!")


@cli.group()
def context():
    """Personal context and preference management."""
    pass


@context.command()
@click.option('--item', required=True, help='Item to record preference for')
@click.option('--type', 'pref_type', type=click.Choice(['like', 'dislike']), required=True)
@click.option('--strength', type=float, default=0.8, help='Strength (0.0-1.0)')
@click.option('--domain', help='Domain (food, transportation, etc.)')
@click.pass_context
def add_preference(ctx, item, pref_type, strength, domain):
    """Record a user preference."""
    vingi = ctx.obj['vingi']
    
    context_data = {}
    if domain:
        context_data['domain'] = domain
    
    pref_id = vingi.context_manager.record_preference(
        item=item,
        preference_type=pref_type,
        strength=strength,
        context=context_data
    )
    
    click.echo(f"✅ Preference recorded: {pref_type} {item} (strength: {strength})")
    vingi.save_all()


@context.command()
@click.option('--domain', help='Filter by domain')
@click.pass_context
def show_preferences(ctx, domain):
    """Show user preferences."""
    vingi = ctx.obj['vingi']
    
    preferences = vingi.context_manager.get_user_preferences(domain)
    
    click.echo(f"📊 User Preferences{f' ({domain})' if domain else ''}")
    click.echo("=" * 30)
    
    if preferences['likes']:
        click.echo("👍 Likes:")
        for like in preferences['likes'][:10]:  # Top 10
            click.echo(f"   • {like['item']} (strength: {like['strength']:.2f})")
    
    if preferences['dislikes']:
        click.echo("\n👎 Dislikes:")
        for dislike in preferences['dislikes'][:10]:  # Top 10
            click.echo(f"   • {dislike['item']} (strength: {dislike['strength']:.2f})")


@context.command()
@click.pass_context
def analyze(ctx):
    """Analyze context graph patterns."""
    vingi = ctx.obj['vingi']
    
    analysis = vingi.context_manager.analyze_patterns()
    
    click.echo("🕸️  Context Graph Analysis")
    click.echo("=" * 30)
    
    click.echo("Node distribution:")
    for node_type, count in analysis['node_distribution'].items():
        click.echo(f"  {node_type}: {count}")
    
    click.echo(f"\nTotal relationships: {analysis['relationship_patterns']['total_relationships']}")
    
    if analysis['preference_trends']['total_preferences'] > 0:
        click.echo(f"Total preferences: {analysis['preference_trends']['total_preferences']}")
        click.echo(f"Likes vs dislikes: {analysis['preference_trends']['likes_vs_dislikes']}")


@cli.group()
def temporal():
    """Temporal pattern analysis and scheduling."""
    pass


@temporal.command()
@click.option('--event-type', required=True, help='Type of event (coding, meeting, etc.)')
@click.option('--duration', type=int, help='Duration in minutes')
@click.option('--energy', type=float, help='Energy level (0.0-1.0)')
@click.option('--focus', type=float, help='Focus level (0.0-1.0)')
@click.option('--complexity', type=click.Choice(['low', 'medium', 'high', 'creative', 'analytical']))
@click.option('--completed', type=bool, help='Whether task was completed')
@click.option('--interruptions', type=int, default=0, help='Number of interruptions')
@click.pass_context
def add_event(ctx, event_type, duration, energy, focus, complexity, completed, interruptions):
    """Add a temporal event for analysis."""
    vingi = ctx.obj['vingi']
    
    event = TemporalEvent(
        timestamp=datetime.now(),
        event_type=event_type,
        duration=timedelta(minutes=duration) if duration else None,
        energy_level=energy,
        focus_level=focus,
        task_complexity=TaskComplexity(complexity) if complexity else None,
        completion_status=completed,
        interruptions=interruptions
    )
    
    vingi.temporal_analyzer.add_event(event)
    click.echo(f"✅ Temporal event recorded: {event_type}")
    vingi.save_all()


@temporal.command()
@click.option('--task-type', required=True, help='Type of task to schedule')
@click.option('--duration', type=int, required=True, help='Required duration in minutes')
@click.option('--days-ahead', type=int, default=7, help='Days ahead to consider')
@click.pass_context
def optimal_times(ctx, task_type, duration, days_ahead):
    """Find optimal time slots for a task."""
    vingi = ctx.obj['vingi']
    
    optimal_slots = vingi.temporal_analyzer.get_optimal_time_slots(
        task_type=task_type,
        duration=timedelta(minutes=duration),
        days_ahead=days_ahead
    )
    
    click.echo(f"⏰ Optimal time slots for {task_type} ({duration} minutes)")
    click.echo("=" * 50)
    
    if optimal_slots:
        for i, slot in enumerate(optimal_slots[:5], 1):
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_name = day_names[slot.day_of_week] if slot.day_of_week is not None else 'Any'
            
            click.echo(f"{i}. {day_name} {slot.start_time} - {slot.end_time}")
            click.echo(f"   Energy: {slot.energy_prediction:.2f} | Focus: {slot.focus_prediction:.2f}")
            click.echo(f"   Confidence: {slot.confidence:.2f}")
            if slot.reasoning:
                click.echo(f"   Reasoning: {', '.join(slot.reasoning)}")
            click.echo()
    else:
        click.echo("No optimal slots found. Try adding more temporal data first.")


@temporal.command()
@click.option('--date', help='Date (YYYY-MM-DD), defaults to tomorrow')
@click.pass_context
def predict_energy(ctx, date):
    """Predict energy levels for a specific date."""
    vingi = ctx.obj['vingi']
    
    if date:
        target_date = datetime.strptime(date, '%Y-%m-%d')
    else:
        target_date = datetime.now() + timedelta(days=1)
    
    predictions = vingi.temporal_analyzer.predict_energy_levels(target_date)
    
    click.echo(f"⚡ Energy predictions for {target_date.strftime('%Y-%m-%d')}")
    click.echo("=" * 40)
    
    # Show top energy hours
    sorted_hours = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    click.echo("Top energy hours:")
    for hour, energy in sorted_hours[:8]:
        click.echo(f"  {hour:02d}:00 - Energy level: {energy:.2f}")


@temporal.command()
@click.pass_context
def analyze_productivity(ctx):
    """Analyze productivity patterns."""
    vingi = ctx.obj['vingi']
    
    analysis = vingi.temporal_analyzer.analyze_productivity_patterns()
    
    if 'error' in analysis:
        click.echo(f"❌ {analysis['error']}")
        return
    
    click.echo("📈 Productivity Analysis")
    click.echo("=" * 30)
    
    # Daily patterns
    if 'daily_patterns' in analysis:
        click.echo("Daily patterns (top hours):")
        daily = analysis['daily_patterns']
        top_hours = sorted([(h, d) for h, d in daily.items() if d['events'] >= 3], 
                          key=lambda x: x[1].get('avg_performance', 0), reverse=True)[:5]
        for hour, data in top_hours:
            click.echo(f"  {hour}:00 - Performance: {data.get('avg_performance', 0):.2f}")
    
    # Weekly patterns
    if 'weekly_patterns' in analysis:
        click.echo("\nWeekly patterns:")
        for day, data in analysis['weekly_patterns'].items():
            click.echo(f"  {day}: {data['avg_performance']:.2f} performance, {data['events']} events")


@cli.group()
def relevance():
    """Information relevance scoring."""
    pass


@relevance.command()
@click.option('--title', help='Information title')
@click.option('--content', required=True, help='Information content')
@click.option('--source', help='Information source')
@click.option('--domain', help='Domain')
@click.option('--tags', help='Comma-separated tags')
@click.pass_context
def score(ctx, title, content, source, domain, tags):
    """Score the relevance of an information item."""
    vingi = ctx.obj['vingi']
    
    # Create information item
    item = InformationItem(
        title=title,
        content=content,
        source=source,
        domain=domain,
        tags=tags.split(',') if tags else [],
        timestamp=datetime.now()
    )
    
    # Create basic user context
    user_context = {
        'active_domains': [domain] if domain else [],
        'time_context': {'urgency_level': 'normal'}
    }
    
    # Score relevance
    score = vingi.relevance_scorer.score_relevance(item, user_context)
    
    click.echo(f"⭐ Relevance Score for: {title or content[:50]}")
    click.echo("=" * 50)
    click.echo(f"Overall Score: {score.overall_score:.2f}")
    click.echo(f"Breakdown:")
    click.echo(f"  Temporal: {score.temporal_score:.2f}")
    click.echo(f"  Personal: {score.personal_score:.2f}")
    click.echo(f"  Contextual: {score.contextual_score:.2f}")
    click.echo(f"  Pattern: {score.pattern_score:.2f}")
    click.echo(f"  Quality: {score.quality_score:.2f}")
    click.echo(f"Confidence: {score.confidence:.2f}")
    
    if score.reasoning:
        click.echo("Reasoning:")
        for reason in score.reasoning:
            click.echo(f"  • {reason}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show overall Vingi system status."""
    vingi = ctx.obj['vingi']
    
    click.echo("🧠 Vingi System Status")
    click.echo("=" * 30)
    
    # Pattern recognizer status
    pattern_summary = vingi.pattern_recognizer.get_pattern_summary()
    click.echo(f"Patterns detected: {pattern_summary['total_patterns_detected']}")
    click.echo(f"Behaviors analyzed: {pattern_summary['behavior_events_analyzed']}")
    
    # Context graph status
    context_stats = vingi.context_manager.get_statistics()
    click.echo(f"Context nodes: {context_stats['node_count']}")
    click.echo(f"Relationships: {context_stats['relationship_count']}")
    
    # Temporal analyzer status
    temporal_stats = vingi.temporal_analyzer.get_statistics()
    click.echo(f"Temporal events: {temporal_stats['total_events']}")
    click.echo(f"Temporal patterns: {temporal_stats['detected_patterns']}")
    
    # Relevance scorer status
    scoring_stats = vingi.relevance_scorer.get_scoring_statistics()
    if isinstance(scoring_stats, dict) and 'total_scored_items' in scoring_stats:
        click.echo(f"Items scored: {scoring_stats['total_scored_items']}")


@cli.command()
@click.option('--export-path', type=click.Path(), help='Path to export data to')
@click.pass_context
def export_data(ctx, export_path):
    """Export all Vingi data."""
    vingi = ctx.obj['vingi']
    
    if not export_path:
        export_path = f"vingi_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Collect all data
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'pattern_summary': vingi.pattern_recognizer.get_pattern_summary(),
        'context_stats': vingi.context_manager.get_statistics(),
        'temporal_stats': vingi.temporal_analyzer.get_statistics(),
        'scoring_stats': vingi.relevance_scorer.get_scoring_statistics()
    }
    
    # Save to file
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    click.echo(f"✅ Data exported to: {export_path}")


@cli.group()
def config():
    """Configuration management."""
    pass


@config.command()
@click.pass_context
def show(ctx):
    """Show current configuration."""
    vingi = ctx.obj['vingi']
    config = vingi.config_manager.config
    
    click.echo("🔧 Vingi Configuration")
    click.echo("=" * 30)
    
    click.echo(f"Data Directory: {config.data_directory}")
    click.echo(f"Log Level: {config.log_level}")
    click.echo(f"Privacy Mode: {config.privacy_mode}")
    click.echo(f"Auto-save Interval: {config.auto_save_interval_minutes} minutes")
    click.echo(f"Intervention Style: {config.intervention_aggressiveness}")
    
    click.echo("\n📊 Pattern Detection:")
    pd_config = config.pattern_detection
    click.echo(f"  Confidence Threshold: {pd_config.confidence_threshold}")
    click.echo(f"  Window Size: {pd_config.window_size}")
    click.echo(f"  Analysis Paralysis Time Multiplier: {pd_config.paralysis_decision_time_multiplier}")
    
    click.echo("\n⭐ Relevance Scoring Weights:")
    rs_config = config.relevance_scoring
    click.echo(f"  Temporal: {rs_config.temporal_weight:.2f}")
    click.echo(f"  Personal: {rs_config.personal_weight:.2f}")
    click.echo(f"  Contextual: {rs_config.contextual_weight:.2f}")
    click.echo(f"  Pattern: {rs_config.pattern_weight:.2f}")
    click.echo(f"  Quality: {rs_config.quality_weight:.2f}")
    
    click.echo("\n⏰ Temporal Analysis:")
    ta_config = config.temporal_analysis
    click.echo(f"  Analysis Window: {ta_config.analysis_window_days} days")
    click.echo(f"  Min Pattern Confidence: {ta_config.min_pattern_confidence}")
    click.echo(f"  Energy Smoothing Window: {ta_config.energy_smoothing_window}")


@config.command()
@click.option('--threshold', type=float, help='Confidence threshold (0.0-1.0)')
@click.option('--window-size', type=int, help='Behavior window size')
@click.option('--paralysis-multiplier', type=float, help='Analysis paralysis time multiplier')
@click.pass_context
def set_patterns(ctx, threshold, window_size, paralysis_multiplier):
    """Update pattern detection settings."""
    vingi = ctx.obj['vingi']
    
    updates = {}
    if threshold is not None:
        updates['confidence_threshold'] = threshold
    if window_size is not None:
        updates['window_size'] = window_size
    if paralysis_multiplier is not None:
        updates['paralysis_decision_time_multiplier'] = paralysis_multiplier
    
    if updates:
        vingi.config_manager.update_pattern_thresholds(**updates)
        click.echo("✅ Pattern detection settings updated")
        
        # Show updated values
        for key, value in updates.items():
            click.echo(f"   {key}: {value}")
    else:
        click.echo("No settings specified to update")


@config.command()
@click.option('--temporal', type=float, help='Temporal weight')
@click.option('--personal', type=float, help='Personal weight')
@click.option('--contextual', type=float, help='Contextual weight')
@click.option('--pattern', type=float, help='Pattern weight')
@click.option('--quality', type=float, help='Quality weight')
@click.pass_context
def set_weights(ctx, temporal, personal, contextual, pattern, quality):
    """Update relevance scoring weights."""
    vingi = ctx.obj['vingi']
    
    updates = {}
    if temporal is not None:
        updates['temporal_weight'] = temporal
    if personal is not None:
        updates['personal_weight'] = personal
    if contextual is not None:
        updates['contextual_weight'] = contextual
    if pattern is not None:
        updates['pattern_weight'] = pattern
    if quality is not None:
        updates['quality_weight'] = quality
    
    if updates:
        vingi.config_manager.update_relevance_weights(**updates)
        click.echo("✅ Relevance weights updated and normalized")
        
        # Show normalized weights
        rs_config = vingi.config_manager.config.relevance_scoring
        click.echo("   Normalized weights:")
        click.echo(f"     Temporal: {rs_config.temporal_weight:.3f}")
        click.echo(f"     Personal: {rs_config.personal_weight:.3f}")
        click.echo(f"     Contextual: {rs_config.contextual_weight:.3f}")
        click.echo(f"     Pattern: {rs_config.pattern_weight:.3f}")
        click.echo(f"     Quality: {rs_config.quality_weight:.3f}")
    else:
        click.echo("No weights specified to update")


@config.command()
@click.option('--analysis-window', type=int, help='Analysis window in days')
@click.option('--min-confidence', type=float, help='Minimum pattern confidence')
@click.option('--smoothing-window', type=int, help='Energy smoothing window size')
@click.pass_context
def set_temporal(ctx, analysis_window, min_confidence, smoothing_window):
    """Update temporal analysis settings."""
    vingi = ctx.obj['vingi']
    
    updates = {}
    if analysis_window is not None:
        updates['analysis_window_days'] = analysis_window
    if min_confidence is not None:
        updates['min_pattern_confidence'] = min_confidence
    if smoothing_window is not None:
        updates['energy_smoothing_window'] = smoothing_window
    
    if updates:
        vingi.config_manager.update_temporal_settings(**updates)
        click.echo("✅ Temporal analysis settings updated")
        
        for key, value in updates.items():
            click.echo(f"   {key}: {value}")
    else:
        click.echo("No settings specified to update")


@config.command()
@click.option('--style', type=click.Choice(['gentle', 'moderate', 'aggressive']), 
              help='Intervention aggressiveness style')
@click.option('--privacy-mode/--no-privacy-mode', help='Enable/disable privacy mode')
@click.option('--auto-save', type=int, help='Auto-save interval in minutes')
@click.pass_context
def set_global(ctx, style, privacy_mode, auto_save):
    """Update global settings."""
    vingi = ctx.obj['vingi']
    config = vingi.config_manager.config
    
    updated = False
    if style is not None:
        config.intervention_aggressiveness = style
        updated = True
        click.echo(f"   Intervention style: {style}")
    
    if privacy_mode is not None:
        config.privacy_mode = privacy_mode
        updated = True
        click.echo(f"   Privacy mode: {privacy_mode}")
    
    if auto_save is not None:
        config.auto_save_interval_minutes = auto_save
        updated = True
        click.echo(f"   Auto-save interval: {auto_save} minutes")
    
    if updated:
        vingi.config_manager.save_config()
        click.echo("✅ Global settings updated")
    else:
        click.echo("No settings specified to update")


@config.command()
@click.pass_context
def validate(ctx):
    """Validate current configuration."""
    vingi = ctx.obj['vingi']
    
    issues = vingi.config_manager.validate_config()
    
    click.echo("🔍 Configuration Validation")
    click.echo("=" * 30)
    
    if issues['errors']:
        click.echo("❌ Errors:")
        for error in issues['errors']:
            click.echo(f"   • {error}")
    
    if issues['warnings']:
        click.echo("\n⚠️  Warnings:")
        for warning in issues['warnings']:
            click.echo(f"   • {warning}")
    
    if issues['suggestions']:
        click.echo("\n💡 Suggestions:")
        for suggestion in issues['suggestions']:
            click.echo(f"   • {suggestion}")
    
    if not any(issues.values()):
        click.echo("✅ Configuration is valid!")


@config.command()
@click.confirmation_option(prompt='Are you sure you want to reset all settings to defaults?')
@click.pass_context
def reset(ctx):
    """Reset configuration to defaults."""
    vingi = ctx.obj['vingi']
    
    vingi.config_manager.reset_to_defaults()
    click.echo("✅ Configuration reset to defaults")


@config.command()
@click.argument('profile_name')
@click.pass_context
def save_profile(ctx, profile_name):
    """Save current configuration as a named profile."""
    vingi = ctx.obj['vingi']
    
    profile_path = vingi.config_manager.create_profile(profile_name)
    click.echo(f"✅ Configuration profile saved: {profile_name}")
    click.echo(f"   Path: {profile_path}")


@config.command()
@click.argument('profile_name')
@click.pass_context
def load_profile(ctx, profile_name):
    """Load a configuration profile."""
    vingi = ctx.obj['vingi']
    
    if vingi.config_manager.load_profile(profile_name):
        click.echo(f"✅ Configuration profile loaded: {profile_name}")
    else:
        click.echo(f"❌ Failed to load profile: {profile_name}")


@config.command()
@click.pass_context
def list_profiles(ctx):
    """List available configuration profiles."""
    vingi = ctx.obj['vingi']
    
    profiles = vingi.config_manager.list_profiles()
    
    click.echo("📁 Available Configuration Profiles")
    click.echo("=" * 40)
    
    if profiles:
        for profile in profiles:
            click.echo(f"   • {profile}")
    else:
        click.echo("   No saved profiles found")


@config.command()
@click.option('--format', type=click.Choice(['yaml', 'json']), default='yaml',
              help='Export format')
@click.argument('export_path', type=click.Path())
@click.pass_context
def export(ctx, format, export_path):
    """Export configuration to a file."""
    vingi = ctx.obj['vingi']
    
    export_path = Path(export_path)
    if vingi.config_manager.export_config(export_path, format):
        click.echo(f"✅ Configuration exported to: {export_path}")
    else:
        click.echo(f"❌ Failed to export configuration")


@config.command()
@click.argument('import_path', type=click.Path(exists=True))
@click.pass_context
def import_config(ctx, import_path):
    """Import configuration from a file."""
    vingi = ctx.obj['vingi']
    
    import_path = Path(import_path)
    if vingi.config_manager.import_config(import_path):
        click.echo(f"✅ Configuration imported from: {import_path}")
        click.echo("   Previous configuration backed up")
    else:
        click.echo(f"❌ Failed to import configuration")


if __name__ == '__main__':
    cli() 