"""
Temporal Pattern Recognition for Cognitive Load Optimization

This module implements pattern detection for the four core cognitive inefficiency patterns:
1. Analysis Paralysis Syndrome
2. Tunnel Vision Planning  
3. Default Behavior Loops
4. Exceptional Ability Self-Doubt
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class CognitivePatternType(Enum):
    """Types of cognitive patterns detected by the system."""
    ANALYSIS_PARALYSIS = "analysis_paralysis"
    TUNNEL_VISION = "tunnel_vision"
    DEFAULT_BEHAVIOR_LOOP = "default_behavior_loop"
    EXCEPTIONAL_ABILITY_DOUBT = "exceptional_ability_doubt"


@dataclass
class PatternDetectionEvent:
    """Represents a single pattern detection event."""
    timestamp: datetime
    pattern_type: CognitivePatternType
    confidence: float  # 0.0 to 1.0
    context: Dict[str, Any]
    severity: str  # "low", "medium", "high"
    intervention_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorData:
    """User behavior data for pattern analysis."""
    timestamp: datetime
    action_type: str
    duration: Optional[timedelta] = None
    decision_count: int = 0
    context_switches: int = 0
    task_complexity: str = "medium"  # "low", "medium", "high"
    domain: str = "general"
    success_indicator: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalPatternRecognizer:
    """
    Advanced pattern recognition engine for detecting cognitive inefficiency patterns
    over time and recommending interventions.
    """
    
    def __init__(self, window_size: int = 100, confidence_threshold: float = 0.7):
        """
        Initialize the pattern recognizer.
        
        Args:
            window_size: Number of recent events to consider for pattern detection
            confidence_threshold: Minimum confidence level for pattern detection
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.behavior_history: deque = deque(maxlen=window_size)
        self.detected_patterns: List[PatternDetectionEvent] = []
        
        # Pattern-specific thresholds and parameters
        self.paralysis_thresholds = {
            'decision_time_multiplier': 3.0,  # 3x normal decision time
            'research_loop_count': 4,  # More than 4 research iterations
            'information_overload_threshold': 10  # More than 10 info sources
        }
        
        self.tunnel_vision_thresholds = {
            'domain_focus_ratio': 0.9,  # 90% of planning effort in one domain
            'neglected_domain_count': 2,  # At least 2 domains ignored
            'planning_imbalance_score': 0.8
        }
        
        self.default_loop_thresholds = {
            'repetition_rate': 0.8,  # 80% of choices are repeats
            'exploration_absence_days': 7,  # No new choices for 7 days
            'constraint_optimization_potential': 0.3  # 30% improvement possible
        }
    
    def add_behavior_event(self, behavior: BehaviorData) -> Optional[PatternDetectionEvent]:
        """
        Add a new behavior event and check for pattern emergence.
        
        Args:
            behavior: New behavior data to analyze
            
        Returns:
            Detected pattern event if any, None otherwise
        """
        self.behavior_history.append(behavior)
        
        # Run pattern detection on the updated history
        detected_pattern = self._analyze_patterns()
        
        if detected_pattern:
            self.detected_patterns.append(detected_pattern)
            logger.info(f"Detected pattern: {detected_pattern.pattern_type.value} "
                       f"with confidence {detected_pattern.confidence:.2f}")
        
        return detected_pattern
    
    def _analyze_patterns(self) -> Optional[PatternDetectionEvent]:
        """
        Analyze recent behavior history for cognitive patterns.
        
        Returns:
            Detected pattern event if confidence exceeds threshold
        """
        if len(self.behavior_history) < 10:  # Need minimum history
            return None
        
        # Check each pattern type
        pattern_analyses = [
            self._detect_analysis_paralysis(),
            self._detect_tunnel_vision(),
            self._detect_default_behavior_loops(),
            self._detect_exceptional_ability_doubt()
        ]
        
        # Return the highest confidence pattern above threshold
        valid_patterns = [p for p in pattern_analyses if p and p.confidence >= self.confidence_threshold]
        
        if valid_patterns:
            return max(valid_patterns, key=lambda p: p.confidence)
        
        return None
    
    def _detect_analysis_paralysis(self) -> Optional[PatternDetectionEvent]:
        """Detect analysis paralysis syndrome."""
        recent_decisions = [b for b in self.behavior_history 
                          if b.action_type in ['research', 'comparison', 'decision']]
        
        if len(recent_decisions) < 5:
            return None
        
        # Calculate analysis metrics
        avg_decision_time = np.mean([b.duration.total_seconds() if b.duration else 0 
                                   for b in recent_decisions])
        research_loops = self._count_research_loops(recent_decisions)
        information_sources = self._count_information_sources(recent_decisions)
        
        # Scoring algorithm
        time_score = min(1.0, avg_decision_time / (30 * 60))  # Normalize to 30 min baseline
        loop_score = min(1.0, research_loops / self.paralysis_thresholds['research_loop_count'])
        info_score = min(1.0, information_sources / self.paralysis_thresholds['information_overload_threshold'])
        
        confidence = (time_score * 0.4 + loop_score * 0.4 + info_score * 0.2)
        
        if confidence >= self.confidence_threshold:
            severity = "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low"
            
            interventions = [
                "Set a maximum research time limit (10-15 minutes)",
                "Use 'good enough' decision criteria",
                "Delegate detailed research to Vingi",
                "Apply time-boxing to decision phases"
            ]
            
            return PatternDetectionEvent(
                timestamp=datetime.now(),
                pattern_type=CognitivePatternType.ANALYSIS_PARALYSIS,
                confidence=confidence,
                context={
                    'avg_decision_time_minutes': avg_decision_time / 60,
                    'research_loops': research_loops,
                    'information_sources': information_sources
                },
                severity=severity,
                intervention_suggestions=interventions
            )
        
        return None
    
    def _detect_tunnel_vision(self) -> Optional[PatternDetectionEvent]:
        """Detect tunnel vision planning pattern."""
        planning_events = [b for b in self.behavior_history 
                         if b.action_type == 'planning']
        
        if len(planning_events) < 3:
            return None
        
        # Analyze domain focus distribution
        domain_distribution = defaultdict(int)
        for event in planning_events:
            domain_distribution[event.domain] += 1
        
        total_planning = len(planning_events)
        if total_planning == 0:
            return None
        
        # Calculate focus concentration
        max_domain_focus = max(domain_distribution.values()) / total_planning
        unique_domains = len(domain_distribution)
        
        # Expected critical domains for comprehensive planning
        critical_domains = ['food', 'transportation', 'accommodation', 'activities', 'safety']
        covered_domains = set(domain_distribution.keys())
        missing_critical = set(critical_domains) - covered_domains
        
        focus_score = max_domain_focus if max_domain_focus > self.tunnel_vision_thresholds['domain_focus_ratio'] else 0
        missing_score = len(missing_critical) / len(critical_domains)
        
        confidence = (focus_score * 0.6 + missing_score * 0.4)
        
        if confidence >= self.confidence_threshold:
            severity = "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low"
            
            interventions = [
                f"Review planning for these neglected domains: {', '.join(missing_critical)}",
                "Use comprehensive planning checklist",
                "Set planning time for each critical domain",
                "Consider backup plans for essential needs"
            ]
            
            return PatternDetectionEvent(
                timestamp=datetime.now(),
                pattern_type=CognitivePatternType.TUNNEL_VISION,
                confidence=confidence,
                context={
                    'max_domain_focus_ratio': max_domain_focus,
                    'covered_domains': list(covered_domains),
                    'missing_critical_domains': list(missing_critical)
                },
                severity=severity,
                intervention_suggestions=interventions
            )
        
        return None
    
    def _detect_default_behavior_loops(self) -> Optional[PatternDetectionEvent]:
        """Detect default behavior loop patterns."""
        choice_events = [b for b in self.behavior_history 
                        if b.action_type in ['choice', 'selection', 'purchase']]
        
        if len(choice_events) < 10:
            return None
        
        # Analyze choice repetition patterns
        choice_patterns = defaultdict(int)
        for event in choice_events:
            choice_key = f"{event.domain}:{event.metadata.get('choice_made', 'unknown')}"
            choice_patterns[choice_key] += 1
        
        total_choices = len(choice_events)
        if total_choices == 0:
            return None
        
        # Calculate repetition metrics
        max_repetition = max(choice_patterns.values())
        repetition_rate = max_repetition / total_choices
        unique_choices = len(choice_patterns)
        
        # Check for exploration absence
        recent_choices = choice_events[-14:]  # Last 2 weeks
        recent_unique = len(set(f"{e.domain}:{e.metadata.get('choice_made', 'unknown')}" 
                               for e in recent_choices))
        exploration_score = 1.0 - (recent_unique / len(recent_choices)) if recent_choices else 1.0
        
        # Calculate optimization potential (simulated)
        optimization_potential = self._estimate_optimization_potential(choice_events)
        
        loop_score = repetition_rate if repetition_rate > self.default_loop_thresholds['repetition_rate'] else 0
        exploration_score = exploration_score if exploration_score > 0.7 else 0
        optimization_score = optimization_potential / self.default_loop_thresholds['constraint_optimization_potential']
        
        confidence = (loop_score * 0.4 + exploration_score * 0.3 + optimization_score * 0.3)
        
        if confidence >= self.confidence_threshold:
            severity = "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low"
            
            # Find the most repeated pattern
            most_repeated_pattern = max(choice_patterns.items(), key=lambda x: x[1])
            
            interventions = [
                f"Try alternatives to your usual: {most_repeated_pattern[0]}",
                "Explore options with similar quality/safety profiles",
                "Consider multi-stop optimization for better choices",
                "Set weekly exploration goals for variety"
            ]
            
            return PatternDetectionEvent(
                timestamp=datetime.now(),
                pattern_type=CognitivePatternType.DEFAULT_BEHAVIOR_LOOP,
                confidence=confidence,
                context={
                    'repetition_rate': repetition_rate,
                    'most_repeated_choice': most_repeated_pattern[0],
                    'repetition_count': most_repeated_pattern[1],
                    'optimization_potential': optimization_potential
                },
                severity=severity,
                intervention_suggestions=interventions
            )
        
        return None
    
    def _detect_exceptional_ability_doubt(self) -> Optional[PatternDetectionEvent]:
        """Detect exceptional ability self-doubt patterns."""
        ability_events = [b for b in self.behavior_history 
                         if b.action_type in ['recall', 'calculation', 'analysis', 'memory_task']]
        
        if len(ability_events) < 5:
            return None
        
        # Analyze performance vs. self-assessment patterns
        actual_performance = [e.success_indicator for e in ability_events if e.success_indicator is not None]
        self_confidence = [e.metadata.get('self_confidence', 0.5) for e in ability_events]
        task_complexity = [e.task_complexity for e in ability_events]
        
        if not actual_performance or not self_confidence:
            return None
        
        # Calculate performance-confidence gap
        avg_performance = np.mean([1.0 if p else 0.0 for p in actual_performance])
        avg_confidence = np.mean(self_confidence)
        confidence_gap = avg_performance - avg_confidence
        
        # Check for underconfidence in high-performance tasks
        high_performance_events = [i for i, p in enumerate(actual_performance) if p]
        underconfidence_in_success = np.mean([self_confidence[i] for i in high_performance_events]) if high_performance_events else 0.5
        
        # Social expectation deviation score
        complex_tasks = [i for i, t in enumerate(task_complexity) if t == 'high']
        complex_performance = np.mean([1.0 if actual_performance[i] else 0.0 for i in complex_tasks]) if complex_tasks else 0.5
        
        gap_score = min(1.0, confidence_gap / 0.5) if confidence_gap > 0.2 else 0
        underconfidence_score = (1.0 - underconfidence_in_success) if underconfidence_in_success < 0.6 else 0
        exceptional_performance_score = complex_performance if complex_performance > 0.8 else 0
        
        confidence = (gap_score * 0.4 + underconfidence_score * 0.3 + exceptional_performance_score * 0.3)
        
        if confidence >= self.confidence_threshold:
            severity = "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low"
            
            interventions = [
                "Track your actual success rates to build confidence",
                "Recognize that complex tasks feeling 'easy' indicates skill, not luck",
                "Maintain your exceptional abilities despite social expectations",
                "Use objective performance metrics over subjective feelings"
            ]
            
            return PatternDetectionEvent(
                timestamp=datetime.now(),
                pattern_type=CognitivePatternType.EXCEPTIONAL_ABILITY_DOUBT,
                confidence=confidence,
                context={
                    'performance_confidence_gap': confidence_gap,
                    'avg_actual_performance': avg_performance,
                    'avg_self_confidence': avg_confidence,
                    'exceptional_performance_rate': complex_performance
                },
                severity=severity,
                intervention_suggestions=interventions
            )
        
        return None
    
    def _count_research_loops(self, decisions: List[BehaviorData]) -> int:
        """Count research loops in decision-making process."""
        research_events = [d for d in decisions if d.action_type == 'research']
        # Group by task/decision context
        task_groups = defaultdict(list)
        for event in research_events:
            task_key = event.metadata.get('task_id', 'unknown')
            task_groups[task_key].append(event)
        
        # Count loops (multiple research events for same task)
        loops = sum(max(0, len(events) - 1) for events in task_groups.values())
        return loops
    
    def _count_information_sources(self, decisions: List[BehaviorData]) -> int:
        """Count distinct information sources consulted."""
        sources = set()
        for decision in decisions:
            source = decision.metadata.get('information_source')
            if source:
                sources.add(source)
        return len(sources)
    
    def _estimate_optimization_potential(self, choice_events: List[BehaviorData]) -> float:
        """Estimate potential improvement from breaking choice constraints."""
        # Simplified heuristic: more repetition = higher optimization potential
        if not choice_events:
            return 0.0
        
        choice_counts = defaultdict(int)
        for event in choice_events:
            choice_key = f"{event.domain}:{event.metadata.get('choice_made', 'unknown')}"
            choice_counts[choice_key] += 1
        
        max_repetition = max(choice_counts.values())
        total_choices = len(choice_events)
        
        # Higher repetition rate suggests more constraint-driven choices
        repetition_rate = max_repetition / total_choices
        return min(1.0, repetition_rate * 1.5)  # Scale to potential improvement
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns."""
        pattern_counts = defaultdict(int)
        recent_patterns = []
        
        for pattern in self.detected_patterns[-20:]:  # Last 20 patterns
            pattern_counts[pattern.pattern_type.value] += 1
            if pattern.timestamp > datetime.now() - timedelta(days=7):
                recent_patterns.append(pattern)
        
        return {
            'total_patterns_detected': len(self.detected_patterns),
            'pattern_type_distribution': dict(pattern_counts),
            'recent_patterns': len(recent_patterns),
            'behavior_events_analyzed': len(self.behavior_history),
            'average_confidence': np.mean([p.confidence for p in self.detected_patterns]) if self.detected_patterns else 0.0
        }
    
    def get_intervention_recommendations(self) -> List[str]:
        """Get current intervention recommendations based on recent patterns."""
        recent_patterns = [p for p in self.detected_patterns 
                          if p.timestamp > datetime.now() - timedelta(days=3)]
        
        if not recent_patterns:
            return ["No recent patterns detected. Continue normal usage."]
        
        # Aggregate interventions by priority
        interventions = []
        for pattern in sorted(recent_patterns, key=lambda p: p.confidence, reverse=True):
            interventions.extend(pattern.intervention_suggestions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_interventions = []
        for intervention in interventions:
            if intervention not in seen:
                seen.add(intervention)
                unique_interventions.append(intervention)
        
        return unique_interventions[:10]  # Top 10 recommendations
