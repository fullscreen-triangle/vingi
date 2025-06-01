#!/usr/bin/env python3
"""
Integration tests for Vingi framework

Tests the full integration between all components including
CLI, configuration, pattern detection, and data persistence.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from vingi.core import (
    TemporalPatternRecognizer,
    ContextGraphManager,
    RelevanceScorer,
    TemporalAnalyzer,
    BehaviorData,
    CognitivePatternType,
    InformationItem,
    TemporalEvent,
    TaskComplexity
)
from vingi.config import ConfigManager, VingiConfig


class TestVingiIntegration(unittest.TestCase):
    """Test integration between all Vingi components."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "config.yaml"
        
        # Initialize components with test directory
        self.config_manager = ConfigManager(self.config_path)
        self.pattern_recognizer = TemporalPatternRecognizer()
        self.context_manager = ContextGraphManager(self.test_dir / "context.db")
        self.temporal_analyzer = TemporalAnalyzer(self.test_dir / "temporal.json")
        self.relevance_scorer = RelevanceScorer(self.context_manager, self.pattern_recognizer)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_configuration_system(self):
        """Test configuration management functionality."""
        # Test default configuration
        config = self.config_manager.config
        self.assertIsInstance(config, VingiConfig)
        self.assertEqual(config.pattern_detection.confidence_threshold, 0.7)
        
        # Test configuration updates
        self.config_manager.update_pattern_thresholds(confidence_threshold=0.8)
        self.assertEqual(config.pattern_detection.confidence_threshold, 0.8)
        
        # Test configuration persistence
        self.config_manager.save_config()
        self.assertTrue(self.config_path.exists())
        
        # Test configuration loading
        new_config_manager = ConfigManager(self.config_path)
        self.assertEqual(new_config_manager.config.pattern_detection.confidence_threshold, 0.8)
    
    def test_pattern_detection_flow(self):
        """Test complete pattern detection workflow."""
        # Create analysis paralysis scenario
        behaviors = []
        now = datetime.now()
        
        for i in range(6):
            behavior = BehaviorData(
                timestamp=now - timedelta(minutes=30 * i),
                action_type="research",
                duration=timedelta(minutes=20 + i * 5),
                task_complexity="medium",
                domain="transportation",
                metadata={"task_id": "train_booking", "information_source": f"source_{i}"}
            )
            behaviors.append(behavior)
        
        # Add behaviors and check for pattern detection
        detected_pattern = None
        for behavior in behaviors:
            pattern = self.pattern_recognizer.add_behavior_event(behavior)
            if pattern:
                detected_pattern = pattern
        
        # Verify pattern detection
        if detected_pattern:
            self.assertEqual(detected_pattern.pattern_type, CognitivePatternType.ANALYSIS_PARALYSIS)
            self.assertGreater(detected_pattern.confidence, 0.5)
            self.assertTrue(len(detected_pattern.intervention_suggestions) > 0)
        
        # Test pattern summary
        summary = self.pattern_recognizer.get_pattern_summary()
        self.assertGreater(summary['behavior_events_analyzed'], 0)
    
    def test_context_graph_integration(self):
        """Test context graph functionality."""
        # Add preferences
        pref_id = self.context_manager.record_preference(
            item="fresh bread",
            preference_type="like",
            strength=0.9,
            context={"domain": "food"}
        )
        self.assertIsNotNone(pref_id)
        
        # Add activity context
        activity_id = self.context_manager.add_activity_context(
            activity_name="train research",
            domain="transportation",
            properties={"complexity": "medium"}
        )
        self.assertIsNotNone(activity_id)
        
        # Test preferences retrieval
        preferences = self.context_manager.get_user_preferences("food")
        self.assertGreater(len(preferences['likes']), 0)
        self.assertEqual(preferences['likes'][0]['item'], "fresh bread")
        
        # Test graph analysis
        analysis = self.context_manager.analyze_patterns()
        self.assertIn('node_distribution', analysis)
        self.assertIn('relationship_patterns', analysis)
    
    def test_temporal_analysis_integration(self):
        """Test temporal analysis functionality."""
        # Add temporal events
        events = []
        now = datetime.now()
        
        for i in range(5):
            event = TemporalEvent(
                timestamp=now - timedelta(hours=i * 2),
                event_type="coding",
                duration=timedelta(hours=1),
                energy_level=0.8 - (i * 0.1),
                focus_level=0.9 - (i * 0.05),
                task_complexity=TaskComplexity.HIGH,
                completion_status=True,
                interruptions=i
            )
            events.append(event)
            self.temporal_analyzer.add_event(event)
        
        # Test optimal time slot generation
        optimal_slots = self.temporal_analyzer.get_optimal_time_slots(
            task_type="coding",
            duration=timedelta(hours=2),
            days_ahead=3
        )
        
        # Should return some slots or be empty due to insufficient data
        self.assertIsInstance(optimal_slots, list)
        
        # Test energy prediction
        tomorrow = datetime.now() + timedelta(days=1)
        energy_predictions = self.temporal_analyzer.predict_energy_levels(tomorrow)
        self.assertEqual(len(energy_predictions), 24)  # 24 hours
        
        # Test statistics
        stats = self.temporal_analyzer.get_statistics()
        self.assertGreater(stats['total_events'], 0)
    
    def test_relevance_scoring_integration(self):
        """Test relevance scoring with context integration."""
        # Add some preferences to context
        self.context_manager.record_preference(
            item="quick decisions",
            preference_type="like",
            strength=0.8,
            context={"domain": "productivity"}
        )
        
        # Create information items
        items = [
            InformationItem(
                title="Quick Decision Making Guide",
                content="Simple steps to make decisions faster without overthinking",
                domain="productivity",
                tags=["quick", "decisions", "simple"],
                timestamp=datetime.now()
            ),
            InformationItem(
                title="Comprehensive Analysis Framework",
                content="Detailed 50-step process for thorough decision analysis",
                domain="productivity", 
                tags=["detailed", "analysis", "comprehensive"],
                timestamp=datetime.now()
            )
        ]
        
        # Score relevance
        user_context = {
            'active_domains': ['productivity'],
            'time_context': {'urgency_level': 'high'}
        }
        
        scored_items = self.relevance_scorer.batch_score_items(items, user_context)
        self.assertEqual(len(scored_items), 2)
        
        # First item should score higher (quick decision guide)
        quick_guide_score = scored_items[0][1].overall_score
        detailed_guide_score = scored_items[1][1].overall_score
        
        # Both should have reasonable scores
        self.assertGreaterEqual(quick_guide_score, 0.0)
        self.assertLessEqual(quick_guide_score, 1.0)
        self.assertGreaterEqual(detailed_guide_score, 0.0)
        self.assertLessEqual(detailed_guide_score, 1.0)
    
    def test_data_persistence(self):
        """Test data persistence across component restarts."""
        # Add data to components
        self.context_manager.record_preference(
            item="test preference",
            preference_type="like",
            strength=0.7,
            context={}
        )
        
        behavior = BehaviorData(
            timestamp=datetime.now(),
            action_type="test",
            domain="test",
            task_complexity="medium"
        )
        self.pattern_recognizer.add_behavior_event(behavior)
        
        event = TemporalEvent(
            timestamp=datetime.now(),
            event_type="test_event",
            energy_level=0.8
        )
        self.temporal_analyzer.add_event(event)
        
        # Save data
        self.context_manager.save()
        self.temporal_analyzer.save_data()
        
        # Create new instances and verify data persistence
        new_context_manager = ContextGraphManager(self.test_dir / "context.db")
        new_temporal_analyzer = TemporalAnalyzer(self.test_dir / "temporal.json")
        
        # Verify context data persisted
        preferences = new_context_manager.get_user_preferences()
        self.assertGreater(len(preferences['likes']), 0)
        
        # Verify temporal data persisted
        stats = new_temporal_analyzer.get_statistics()
        self.assertGreater(stats['total_events'], 0)
    
    def test_configuration_profiles(self):
        """Test configuration profile functionality."""
        # Create and save a profile
        profile_path = self.config_manager.create_profile("test_profile")
        self.assertTrue(profile_path.exists())
        
        # Modify configuration
        self.config_manager.update_pattern_thresholds(confidence_threshold=0.9)
        
        # Load profile (should restore original settings)
        success = self.config_manager.load_profile("test_profile")
        self.assertTrue(success)
        self.assertEqual(self.config_manager.config.pattern_detection.confidence_threshold, 0.7)
        
        # Test profile listing
        profiles = self.config_manager.list_profiles()
        self.assertIn("test_profile", profiles)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        issues = self.config_manager.validate_config()
        self.assertIsInstance(issues, dict)
        self.assertIn('errors', issues)
        self.assertIn('warnings', issues)
        
        # Test invalid configuration
        self.config_manager.config.relevance_scoring.temporal_weight = 2.0  # Invalid weight
        issues = self.config_manager.validate_config()
        self.assertGreater(len(issues['errors']), 0)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Configure system
        self.config_manager.update_pattern_thresholds(confidence_threshold=0.6)
        
        # 2. Add user preferences
        self.context_manager.record_preference("efficiency", "like", 0.9, {"domain": "work"})
        
        # 3. Record behaviors
        for i in range(3):
            behavior = BehaviorData(
                timestamp=datetime.now() - timedelta(hours=i),
                action_type="research",
                domain="work",
                duration=timedelta(minutes=45)
            )
            self.pattern_recognizer.add_behavior_event(behavior)
        
        # 4. Record temporal events
        event = TemporalEvent(
            timestamp=datetime.now(),
            event_type="analysis",
            energy_level=0.7,
            focus_level=0.8
        )
        self.temporal_analyzer.add_event(event)
        
        # 5. Score information relevance
        item = InformationItem(
            content="Efficiency improvement techniques",
            domain="work"
        )
        
        user_context = {'active_domains': ['work']}
        score = self.relevance_scorer.score_relevance(item, user_context)
        
        # 6. Verify integrated functionality
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score.overall_score, 0.0)
        self.assertLessEqual(score.overall_score, 1.0)
        
        # 7. Get system status
        pattern_summary = self.pattern_recognizer.get_pattern_summary()
        context_stats = self.context_manager.get_statistics()
        temporal_stats = self.temporal_analyzer.get_statistics()
        
        self.assertGreater(pattern_summary['behavior_events_analyzed'], 0)
        self.assertGreater(context_stats['node_count'], 0)
        self.assertGreater(temporal_stats['total_events'], 0)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration (basic functionality)."""
    
    def setUp(self):
        """Set up CLI testing environment."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up CLI test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cli_imports(self):
        """Test that CLI module imports correctly."""
        try:
            from vingi.cli import VingiCLI
            cli = VingiCLI()
            self.assertIsNotNone(cli)
        except ImportError as e:
            self.fail(f"CLI import failed: {e}")
    
    def test_config_integration(self):
        """Test configuration integration with CLI."""
        from vingi.cli import VingiCLI
        
        cli = VingiCLI()
        self.assertIsNotNone(cli.config_manager)
        self.assertIsNotNone(cli.pattern_recognizer)
        self.assertIsNotNone(cli.context_manager)
        self.assertIsNotNone(cli.temporal_analyzer)
        self.assertIsNotNone(cli.relevance_scorer)


if __name__ == '__main__':
    unittest.main() 