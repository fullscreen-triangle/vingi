"""
Vingi Configuration Management

Provides centralized configuration for all Vingi components with
customizable thresholds, weights, and behavioral parameters.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternDetectionConfig:
    """Configuration for pattern detection algorithms."""
    confidence_threshold: float = 0.7
    window_size: int = 100
    
    # Analysis Paralysis thresholds
    paralysis_decision_time_multiplier: float = 3.0
    paralysis_research_loop_count: int = 4
    paralysis_info_overload_threshold: int = 10
    
    # Tunnel Vision thresholds
    tunnel_domain_focus_ratio: float = 0.9
    tunnel_neglected_domain_count: int = 2
    tunnel_planning_imbalance_score: float = 0.8
    
    # Default Behavior Loop thresholds
    default_repetition_rate: float = 0.8
    default_exploration_absence_days: int = 7
    default_constraint_optimization_potential: float = 0.3


@dataclass
class RelevanceScoringConfig:
    """Configuration for relevance scoring algorithms."""
    temporal_weight: float = 0.25
    personal_weight: float = 0.30
    contextual_weight: float = 0.20
    pattern_weight: float = 0.15
    quality_weight: float = 0.10
    
    # Learning parameters
    weight_adjustment_factor: float = 0.05
    feedback_prediction_error_threshold: float = 0.3
    
    # Content analysis
    urgency_boost: float = 0.4
    temporal_boost: float = 0.3
    quality_boost: float = 0.3
    quality_penalty: float = 0.4


@dataclass
class TemporalAnalysisConfig:
    """Configuration for temporal analysis algorithms."""
    min_pattern_confidence: float = 0.6
    analysis_window_days: int = 30
    energy_smoothing_window: int = 5
    
    # Performance calculation weights
    energy_weight: float = 0.3
    focus_weight: float = 0.3
    completion_weight: float = 0.2
    complexity_match_weight: float = 0.2
    
    # Time slot optimization
    min_data_points: int = 3
    performance_threshold: float = 0.6
    max_optimal_slots: int = 10
    
    # Energy prediction defaults
    morning_peak_energy: float = 0.8
    afternoon_dip_energy: float = 0.5
    evening_decline_energy: float = 0.4


@dataclass
class ContextGraphConfig:
    """Configuration for context graph management."""
    max_nodes: int = 50000
    max_relationships: int = 100000
    cache_ttl_minutes: int = 5
    
    # Graph optimization
    relevance_decay_rate: float = 0.01  # 1% per day
    min_relevance_threshold: float = 0.1
    max_node_age_days: int = 365
    
    # Query performance
    max_query_results: int = 100
    max_traversal_depth: int = 3


@dataclass
class VingiConfig:
    """Master configuration for Vingi framework."""
    # Component configurations
    pattern_detection: PatternDetectionConfig = field(default_factory=PatternDetectionConfig)
    relevance_scoring: RelevanceScoringConfig = field(default_factory=RelevanceScoringConfig)
    temporal_analysis: TemporalAnalysisConfig = field(default_factory=TemporalAnalysisConfig)
    context_graph: ContextGraphConfig = field(default_factory=ContextGraphConfig)
    
    # Global settings
    data_directory: str = "~/Library/Application Support/Vingi"
    log_level: str = "INFO"
    auto_save_interval_minutes: int = 30
    privacy_mode: bool = True
    
    # User preferences
    user_timezone: str = "UTC"
    preferred_notification_times: list = field(default_factory=lambda: [9, 14, 18])
    intervention_aggressiveness: str = "moderate"  # "gentle", "moderate", "aggressive"


class ConfigManager:
    """
    Manages Vingi configuration with support for multiple formats
    and environment-specific overrides.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file, defaults to user config directory
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._config = VingiConfig()
        self._load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        return Path.home() / "Library" / "Application Support" / "Vingi" / "config.yaml"
    
    def _load_config(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            # Create default config file
            self.save_config()
            logger.info(f"Created default configuration at {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f)
            
            # Update configuration from file
            self._update_config_from_dict(config_data)
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary data."""
        if 'pattern_detection' in config_data:
            self._update_dataclass(self._config.pattern_detection, config_data['pattern_detection'])
        
        if 'relevance_scoring' in config_data:
            self._update_dataclass(self._config.relevance_scoring, config_data['relevance_scoring'])
        
        if 'temporal_analysis' in config_data:
            self._update_dataclass(self._config.temporal_analysis, config_data['temporal_analysis'])
        
        if 'context_graph' in config_data:
            self._update_dataclass(self._config.context_graph, config_data['context_graph'])
        
        # Update global settings
        for key, value in config_data.items():
            if hasattr(self._config, key) and key not in ['pattern_detection', 'relevance_scoring', 'temporal_analysis', 'context_graph']:
                setattr(self._config, key, value)
    
    def _update_dataclass(self, dataclass_instance, update_dict: Dict[str, Any]):
        """Update dataclass fields from dictionary."""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            config_dict = asdict(self._config)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    @property
    def config(self) -> VingiConfig:
        """Get current configuration."""
        return self._config
    
    def get_pattern_detection_config(self) -> PatternDetectionConfig:
        """Get pattern detection configuration."""
        return self._config.pattern_detection
    
    def get_relevance_scoring_config(self) -> RelevanceScoringConfig:
        """Get relevance scoring configuration."""
        return self._config.relevance_scoring
    
    def get_temporal_analysis_config(self) -> TemporalAnalysisConfig:
        """Get temporal analysis configuration."""
        return self._config.temporal_analysis
    
    def get_context_graph_config(self) -> ContextGraphConfig:
        """Get context graph configuration."""
        return self._config.context_graph
    
    def update_pattern_thresholds(self, **kwargs):
        """Update pattern detection thresholds."""
        for key, value in kwargs.items():
            if hasattr(self._config.pattern_detection, key):
                setattr(self._config.pattern_detection, key, value)
        self.save_config()
    
    def update_relevance_weights(self, **kwargs):
        """Update relevance scoring weights."""
        for key, value in kwargs.items():
            if hasattr(self._config.relevance_scoring, key):
                setattr(self._config.relevance_scoring, key, value)
        
        # Normalize weights
        total_weight = (
            self._config.relevance_scoring.temporal_weight +
            self._config.relevance_scoring.personal_weight +
            self._config.relevance_scoring.contextual_weight +
            self._config.relevance_scoring.pattern_weight +
            self._config.relevance_scoring.quality_weight
        )
        
        if total_weight > 0:
            self._config.relevance_scoring.temporal_weight /= total_weight
            self._config.relevance_scoring.personal_weight /= total_weight
            self._config.relevance_scoring.contextual_weight /= total_weight
            self._config.relevance_scoring.pattern_weight /= total_weight
            self._config.relevance_scoring.quality_weight /= total_weight
        
        self.save_config()
    
    def update_temporal_settings(self, **kwargs):
        """Update temporal analysis settings."""
        for key, value in kwargs.items():
            if hasattr(self._config.temporal_analysis, key):
                setattr(self._config.temporal_analysis, key, value)
        self.save_config()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self._config = VingiConfig()
        self.save_config()
        logger.info("Configuration reset to defaults")
    
    def create_profile(self, profile_name: str) -> Path:
        """Create a new configuration profile."""
        profile_path = self.config_path.parent / f"config_{profile_name}.yaml"
        
        # Save current config as new profile
        config_dict = asdict(self._config)
        with open(profile_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created configuration profile: {profile_name}")
        return profile_path
    
    def load_profile(self, profile_name: str) -> bool:
        """Load a configuration profile."""
        profile_path = self.config_path.parent / f"config_{profile_name}.yaml"
        
        if not profile_path.exists():
            logger.error(f"Profile not found: {profile_name}")
            return False
        
        try:
            with open(profile_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            self._update_config_from_dict(config_data)
            logger.info(f"Loaded configuration profile: {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading profile {profile_name}: {e}")
            return False
    
    def list_profiles(self) -> list:
        """List available configuration profiles."""
        profile_files = list(self.config_path.parent.glob("config_*.yaml"))
        profiles = []
        
        for profile_file in profile_files:
            profile_name = profile_file.stem.replace("config_", "")
            profiles.append(profile_name)
        
        return sorted(profiles)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration and return any issues."""
        issues = {
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Validate pattern detection config
        pd_config = self._config.pattern_detection
        if pd_config.confidence_threshold < 0.5:
            issues['warnings'].append("Pattern detection confidence threshold is very low")
        if pd_config.confidence_threshold > 0.95:
            issues['warnings'].append("Pattern detection confidence threshold may be too strict")
        
        # Validate relevance scoring weights
        rs_config = self._config.relevance_scoring
        weight_sum = (rs_config.temporal_weight + rs_config.personal_weight + 
                     rs_config.contextual_weight + rs_config.pattern_weight + rs_config.quality_weight)
        if abs(weight_sum - 1.0) > 0.01:
            issues['errors'].append("Relevance scoring weights do not sum to 1.0")
        
        # Validate temporal analysis config
        ta_config = self._config.temporal_analysis
        if ta_config.analysis_window_days < 7:
            issues['warnings'].append("Temporal analysis window may be too short for reliable patterns")
        
        # Validate context graph config
        cg_config = self._config.context_graph
        if cg_config.max_nodes < 1000:
            issues['warnings'].append("Context graph node limit may be too restrictive")
        
        return issues
    
    def export_config(self, export_path: Path, format: str = 'yaml') -> bool:
        """Export configuration to a file."""
        try:
            config_dict = asdict(self._config)
            
            with open(export_path, 'w') as f:
                if format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, import_path: Path) -> bool:
        """Import configuration from a file."""
        if not import_path.exists():
            logger.error(f"Import file not found: {import_path}")
            return False
        
        try:
            with open(import_path, 'r') as f:
                if import_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f)
            
            # Backup current config
            backup_path = self.config_path.with_suffix('.backup.yaml')
            self.export_config(backup_path, 'yaml')
            
            # Load new config
            self._update_config_from_dict(config_data)
            self.save_config()
            
            logger.info(f"Configuration imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> VingiConfig:
    """Get current Vingi configuration."""
    return get_config_manager().config


def update_config(**kwargs):
    """Update global configuration."""
    config_manager = get_config_manager()
    
    # Route updates to appropriate sections
    for key, value in kwargs.items():
        if key.startswith('pattern_'):
            config_manager.update_pattern_thresholds(**{key[8:]: value})
        elif key.startswith('relevance_'):
            config_manager.update_relevance_weights(**{key[10:]: value})
        elif key.startswith('temporal_'):
            config_manager.update_temporal_settings(**{key[9:]: value})
        else:
            # Global setting
            if hasattr(config_manager.config, key):
                setattr(config_manager.config, key, value)
    
    config_manager.save_config() 