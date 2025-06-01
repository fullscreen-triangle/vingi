"""
Core cognitive load optimization components for Vingi.

This module contains the foundational components for detecting and addressing
cognitive inefficiency patterns.
"""

from .pattern_recognition import (
    TemporalPatternRecognizer,
    CognitivePatternType,
    PatternDetectionEvent,
    BehaviorData
)

from .context_graph import (
    ContextGraphManager,
    ContextNode,
    ContextRelationship,
    ContextQuery,
    NodeType,
    RelationshipType
)

from .relevance_scoring import (
    RelevanceScorer,
    RelevanceScore,
    InformationItem,
    ScoringWeights,
    RelevanceContext
)

from .temporal_analysis import (
    TemporalAnalyzer,
    TemporalEvent,
    TemporalPattern,
    OptimalTimeSlot,
    TemporalPatternType,
    TaskComplexity
)

__all__ = [
    # Pattern Recognition
    "TemporalPatternRecognizer",
    "CognitivePatternType", 
    "PatternDetectionEvent",
    "BehaviorData",
    
    # Context Graph
    "ContextGraphManager",
    "ContextNode",
    "ContextRelationship", 
    "ContextQuery",
    "NodeType",
    "RelationshipType",
    
    # Relevance Scoring
    "RelevanceScorer",
    "RelevanceScore",
    "InformationItem",
    "ScoringWeights",
    "RelevanceContext",
    
    # Temporal Analysis
    "TemporalAnalyzer",
    "TemporalEvent",
    "TemporalPattern",
    "OptimalTimeSlot",
    "TemporalPatternType",
    "TaskComplexity",
]
