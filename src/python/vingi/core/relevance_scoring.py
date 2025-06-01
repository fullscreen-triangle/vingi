"""
Relevance Scoring Engine for Information Filtering

This module implements advanced relevance scoring algorithms to filter and rank
information based on user context, preferences, and cognitive patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict
import re
import math

logger = logging.getLogger(__name__)


class RelevanceContext(Enum):
    """Types of relevance contexts."""
    IMMEDIATE = "immediate"  # Immediate task relevance
    TEMPORAL = "temporal"   # Time-sensitive relevance  
    PERSONAL = "personal"   # Personal preference relevance
    PATTERN = "pattern"     # Cognitive pattern relevance
    DOMAIN = "domain"       # Domain-specific relevance
    SOCIAL = "social"       # Social context relevance


@dataclass
class ScoringWeights:
    """Weights for different relevance factors."""
    temporal_weight: float = 0.25
    personal_weight: float = 0.30
    contextual_weight: float = 0.20
    pattern_weight: float = 0.15
    quality_weight: float = 0.10
    
    def normalize(self):
        """Normalize weights to sum to 1.0."""
        total = (self.temporal_weight + self.personal_weight + 
                self.contextual_weight + self.pattern_weight + self.quality_weight)
        if total > 0:
            self.temporal_weight /= total
            self.personal_weight /= total
            self.contextual_weight /= total
            self.pattern_weight /= total
            self.quality_weight /= total


@dataclass
class RelevanceScore:
    """Represents a relevance score with breakdown."""
    overall_score: float
    temporal_score: float
    personal_score: float
    contextual_score: float
    pattern_score: float
    quality_score: float
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InformationItem:
    """Represents an item to be scored for relevance."""
    content: str
    title: Optional[str] = None
    source: Optional[str] = None
    timestamp: Optional[datetime] = None
    domain: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RelevanceScorer:
    """
    Advanced relevance scoring engine that evaluates information relevance
    based on multiple contextual factors and user patterns.
    """
    
    def __init__(self, context_manager: 'ContextGraphManager' = None,
                 pattern_recognizer: 'TemporalPatternRecognizer' = None):
        """
        Initialize the relevance scorer.
        
        Args:
            context_manager: Context graph manager for user context
            pattern_recognizer: Pattern recognizer for cognitive patterns
        """
        self.context_manager = context_manager
        self.pattern_recognizer = pattern_recognizer
        
        # Default scoring weights
        self.weights = ScoringWeights()
        self.weights.normalize()
        
        # Scoring history for learning
        self.scoring_history: List[Tuple[InformationItem, RelevanceScore, float]] = []
        self.user_feedback_history: Dict[str, float] = {}
        
        # Common patterns for text analysis
        self.urgency_patterns = [
            r'\b(urgent|asap|immediate|now|today|deadline|due)\b',
            r'\b(emergency|critical|important|priority)\b'
        ]
        
        self.temporal_patterns = [
            r'\b(today|tomorrow|this week|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(\d{1,2}:\d{2}|\d{1,2}(am|pm))\b',
            r'\b(\d{1,2}/\d{1,2}|\d{4}-\d{2}-\d{2})\b'
        ]
        
        # Quality indicators
        self.quality_indicators = {
            'positive': ['verified', 'trusted', 'official', 'expert', 'reviewed', 'accurate'],
            'negative': ['spam', 'clickbait', 'unverified', 'rumor', 'fake', 'misleading']
        }
    
    def score_relevance(self, item: InformationItem, 
                       user_context: Dict[str, Any]) -> RelevanceScore:
        """
        Score the relevance of an information item.
        
        Args:
            item: The information item to score
            user_context: Current user context
            
        Returns:
            RelevanceScore object with detailed breakdown
        """
        # Individual score components
        temporal_score = self._score_temporal_relevance(item, user_context)
        personal_score = self._score_personal_relevance(item, user_context)
        contextual_score = self._score_contextual_relevance(item, user_context)
        pattern_score = self._score_pattern_relevance(item, user_context)
        quality_score = self._score_quality(item)
        
        # Calculate weighted overall score
        overall_score = (
            temporal_score * self.weights.temporal_weight +
            personal_score * self.weights.personal_weight +
            contextual_score * self.weights.contextual_weight +
            pattern_score * self.weights.pattern_weight +
            quality_score * self.weights.quality_weight
        )
        
        # Calculate confidence based on available information
        confidence = self._calculate_confidence(item, user_context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            item, temporal_score, personal_score, contextual_score, 
            pattern_score, quality_score
        )
        
        score = RelevanceScore(
            overall_score=overall_score,
            temporal_score=temporal_score,
            personal_score=personal_score,
            contextual_score=contextual_score,
            pattern_score=pattern_score,
            quality_score=quality_score,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'item_id': id(item),
                'scoring_time': datetime.now().isoformat(),
                'context_keys': list(user_context.keys())
            }
        )
        
        # Store in history for learning
        self.scoring_history.append((item, score, confidence))
        
        return score
    
    def _score_temporal_relevance(self, item: InformationItem, 
                                 user_context: Dict[str, Any]) -> float:
        """Score temporal relevance of the item."""
        score = 0.5  # Base score
        
        # Check for urgency indicators
        content_text = f"{item.title or ''} {item.content}".lower()
        urgency_matches = sum(len(re.findall(pattern, content_text)) 
                            for pattern in self.urgency_patterns)
        if urgency_matches > 0:
            score += min(0.4, urgency_matches * 0.1)
        
        # Check for temporal references
        temporal_matches = sum(len(re.findall(pattern, content_text)) 
                             for pattern in self.temporal_patterns)
        if temporal_matches > 0:
            score += min(0.3, temporal_matches * 0.05)
        
        # Item age relevance
        if item.timestamp:
            age_hours = (datetime.now() - item.timestamp).total_seconds() / 3600
            if age_hours < 1:
                score += 0.2  # Very recent
            elif age_hours < 24:
                score += 0.1  # Recent
            elif age_hours > 168:  # Older than a week
                score -= 0.2
        
        # Current time context
        current_time_context = user_context.get('time_context', {})
        if current_time_context.get('urgency_level', 'normal') == 'high':
            score *= 1.2  # Boost during high urgency periods
        
        return max(0.0, min(1.0, score))
    
    def _score_personal_relevance(self, item: InformationItem, 
                                 user_context: Dict[str, Any]) -> float:
        """Score personal relevance based on user preferences."""
        if not self.context_manager:
            return 0.5  # Default if no context available
        
        score = 0.5
        
        # Get user preferences
        preferences = self.context_manager.get_user_preferences(item.domain)
        
        # Check against likes
        likes = preferences.get('likes', [])
        for like in likes:
            like_item = like['item'].lower()
            content_text = f"{item.title or ''} {item.content}".lower()
            if like_item in content_text:
                score += like['strength'] * 0.3
        
        # Check against dislikes
        dislikes = preferences.get('dislikes', [])
        for dislike in dislikes:
            dislike_item = dislike['item'].lower()
            content_text = f"{item.title or ''} {item.content}".lower()
            if dislike_item in content_text:
                score -= dislike['strength'] * 0.4
        
        # Domain preferences
        user_domains = user_context.get('active_domains', [])
        if item.domain and item.domain in user_domains:
            score += 0.2
        
        # Historical interaction patterns
        historical_score = self._get_historical_preference_score(item)
        score += historical_score * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_contextual_relevance(self, item: InformationItem, 
                                   user_context: Dict[str, Any]) -> float:
        """Score contextual relevance based on current user context."""
        score = 0.5
        
        # Current task context
        current_task = user_context.get('current_task')
        if current_task:
            task_keywords = current_task.get('keywords', [])
            content_text = f"{item.title or ''} {item.content}".lower()
            
            keyword_matches = sum(1 for keyword in task_keywords 
                                if keyword.lower() in content_text)
            if task_keywords:
                score += (keyword_matches / len(task_keywords)) * 0.4
        
        # Location context
        current_location = user_context.get('location')
        if current_location and item.metadata.get('location'):
            if current_location == item.metadata['location']:
                score += 0.2
        
        # Social context
        current_contacts = user_context.get('active_contacts', [])
        item_contacts = item.metadata.get('mentioned_contacts', [])
        contact_overlap = set(current_contacts) & set(item_contacts)
        if contact_overlap:
            score += min(0.3, len(contact_overlap) * 0.1)
        
        # Tag relevance
        user_tags = user_context.get('active_tags', [])
        item_tags = set(item.tags)
        tag_overlap = set(user_tags) & item_tags
        if tag_overlap and user_tags:
            score += (len(tag_overlap) / len(user_tags)) * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _score_pattern_relevance(self, item: InformationItem, 
                                user_context: Dict[str, Any]) -> float:
        """Score relevance based on detected cognitive patterns."""
        if not self.pattern_recognizer:
            return 0.5
        
        score = 0.5
        
        # Check recent patterns
        recent_patterns = getattr(self.pattern_recognizer, 'detected_patterns', [])
        if not recent_patterns:
            return score
        
        # Get the most recent pattern
        latest_pattern = recent_patterns[-1] if recent_patterns else None
        if not latest_pattern:
            return score
        
        content_text = f"{item.title or ''} {item.content}".lower()
        
        # Pattern-specific scoring
        if latest_pattern.pattern_type.value == 'analysis_paralysis':
            # Boost quick decision aids, reduce detailed analysis
            if any(word in content_text for word in ['quick', 'simple', 'summary', 'tldr']):
                score += 0.3
            if any(word in content_text for word in ['detailed', 'comprehensive', 'analysis', 'compare']):
                score -= 0.2
        
        elif latest_pattern.pattern_type.value == 'tunnel_vision':
            # Boost comprehensive planning aids
            if any(word in content_text for word in ['checklist', 'comprehensive', 'complete', 'all']):
                score += 0.3
            # Boost neglected domains
            neglected_domains = latest_pattern.context.get('missing_critical_domains', [])
            if item.domain in neglected_domains:
                score += 0.4
        
        elif latest_pattern.pattern_type.value == 'default_behavior_loop':
            # Boost exploration and alternatives
            if any(word in content_text for word in ['alternative', 'new', 'different', 'explore', 'try']):
                score += 0.3
            # Check if it relates to the repeated pattern
            repeated_choice = latest_pattern.context.get('most_repeated_choice', '')
            if repeated_choice.lower() in content_text:
                score += 0.2
        
        elif latest_pattern.pattern_type.value == 'exceptional_ability_doubt':
            # Boost confidence-building content
            if any(word in content_text for word in ['confidence', 'skill', 'ability', 'trust', 'capable']):
                score += 0.3
        
        # Factor in pattern confidence
        pattern_confidence = latest_pattern.confidence
        score = 0.5 + (score - 0.5) * pattern_confidence
        
        return max(0.0, min(1.0, score))
    
    def _score_quality(self, item: InformationItem) -> float:
        """Score the quality of the information item."""
        score = 0.5
        
        content_text = f"{item.title or ''} {item.content} {item.source or ''}".lower()
        
        # Positive quality indicators
        positive_count = sum(1 for indicator in self.quality_indicators['positive']
                           if indicator in content_text)
        score += min(0.3, positive_count * 0.1)
        
        # Negative quality indicators
        negative_count = sum(1 for indicator in self.quality_indicators['negative']
                           if indicator in content_text)
        score -= min(0.4, negative_count * 0.15)
        
        # Source quality
        if item.source:
            trusted_sources = ['gov', 'edu', 'official', 'verified']
            if any(source in item.source.lower() for source in trusted_sources):
                score += 0.2
        
        # Content length and structure (heuristic for quality)
        if item.content:
            word_count = len(item.content.split())
            if 50 <= word_count <= 500:  # Good length range
                score += 0.1
            elif word_count < 10:  # Too short
                score -= 0.2
            elif word_count > 1000:  # Too long
                score -= 0.1
        
        # Title quality
        if item.title:
            title_length = len(item.title.split())
            if 3 <= title_length <= 12:  # Good title length
                score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _calculate_confidence(self, item: InformationItem, 
                             user_context: Dict[str, Any]) -> float:
        """Calculate confidence in the relevance score."""
        confidence = 0.5
        
        # More information available = higher confidence
        available_fields = sum([
            1 if item.title else 0,
            1 if item.source else 0,
            1 if item.timestamp else 0,
            1 if item.domain else 0,
            1 if item.tags else 0,
            1 if item.metadata else 0
        ])
        confidence += (available_fields / 6) * 0.3
        
        # Context availability
        context_richness = len(user_context) / 10.0  # Assume 10 is rich context
        confidence += min(0.3, context_richness * 0.3)
        
        # Historical data availability
        if self.context_manager and len(self.context_manager.nodes) > 100:
            confidence += 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(self, item: InformationItem, temporal: float, 
                          personal: float, contextual: float, 
                          pattern: float, quality: float) -> List[str]:
        """Generate human-readable reasoning for the score."""
        reasoning = []
        
        # Temporal reasoning
        if temporal > 0.7:
            reasoning.append("High temporal relevance - urgent or time-sensitive content")
        elif temporal < 0.3:
            reasoning.append("Low temporal relevance - not time-sensitive")
        
        # Personal reasoning
        if personal > 0.7:
            reasoning.append("High personal relevance - matches your preferences")
        elif personal < 0.3:
            reasoning.append("Low personal relevance - conflicts with your preferences")
        
        # Contextual reasoning
        if contextual > 0.7:
            reasoning.append("High contextual relevance - related to current activities")
        elif contextual < 0.3:
            reasoning.append("Low contextual relevance - not related to current context")
        
        # Pattern reasoning
        if pattern > 0.7:
            reasoning.append("Relevant to detected cognitive patterns - helpful for current challenges")
        elif pattern < 0.3:
            reasoning.append("May not address current cognitive patterns")
        
        # Quality reasoning
        if quality > 0.7:
            reasoning.append("High quality content from trusted sources")
        elif quality < 0.3:
            reasoning.append("Quality concerns - verify information carefully")
        
        return reasoning
    
    def _get_historical_preference_score(self, item: InformationItem) -> float:
        """Get historical preference score for similar items."""
        if not self.user_feedback_history:
            return 0.0
        
        # Simple content similarity based on keywords
        item_words = set((item.title or '').lower().split() + item.content.lower().split())
        
        similar_scores = []
        for content_key, feedback_score in self.user_feedback_history.items():
            key_words = set(content_key.lower().split())
            overlap = len(item_words & key_words)
            if overlap > 2:  # Some similarity threshold
                similar_scores.append(feedback_score)
        
        return np.mean(similar_scores) if similar_scores else 0.0
    
    def record_user_feedback(self, item: InformationItem, 
                           feedback_score: float, 
                           feedback_type: str = 'relevance'):
        """
        Record user feedback on relevance scoring accuracy.
        
        Args:
            item: The item that was scored
            feedback_score: User feedback score (0.0 to 1.0)
            feedback_type: Type of feedback ('relevance', 'quality', etc.)
        """
        content_key = f"{item.title or ''} {item.content[:50]}"
        self.user_feedback_history[content_key] = feedback_score
        
        # Update scoring weights based on feedback
        self._update_weights_from_feedback(item, feedback_score)
        
        logger.info(f"Recorded user feedback: {feedback_score} for {feedback_type}")
    
    def _update_weights_from_feedback(self, item: InformationItem, feedback_score: float):
        """Update scoring weights based on user feedback."""
        # Find the corresponding score in history
        for hist_item, hist_score, _ in self.scoring_history[-10:]:  # Last 10 scores
            if id(hist_item) == id(item):
                # Simple weight adjustment based on feedback accuracy
                prediction_error = abs(hist_score.overall_score - feedback_score)
                
                # If prediction was off, adjust weights
                if prediction_error > 0.3:
                    adjustment_factor = 0.05  # Small adjustments
                    
                    # Adjust weights based on which components were most off
                    if abs(hist_score.temporal_score - feedback_score) > 0.3:
                        self.weights.temporal_weight *= (1 - adjustment_factor)
                    if abs(hist_score.personal_score - feedback_score) > 0.3:
                        self.weights.personal_weight *= (1 - adjustment_factor)
                    if abs(hist_score.contextual_score - feedback_score) > 0.3:
                        self.weights.contextual_weight *= (1 - adjustment_factor)
                    if abs(hist_score.pattern_score - feedback_score) > 0.3:
                        self.weights.pattern_weight *= (1 - adjustment_factor)
                    if abs(hist_score.quality_score - feedback_score) > 0.3:
                        self.weights.quality_weight *= (1 - adjustment_factor)
                    
                    # Renormalize weights
                    self.weights.normalize()
                break
    
    def batch_score_items(self, items: List[InformationItem], 
                         user_context: Dict[str, Any]) -> List[Tuple[InformationItem, RelevanceScore]]:
        """
        Score multiple items in batch for efficiency.
        
        Args:
            items: List of items to score
            user_context: Current user context
            
        Returns:
            List of (item, score) tuples sorted by relevance
        """
        scored_items = []
        
        for item in items:
            score = self.score_relevance(item, user_context)
            scored_items.append((item, score))
        
        # Sort by overall score descending
        scored_items.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return scored_items
    
    def get_top_relevant_items(self, items: List[InformationItem], 
                              user_context: Dict[str, Any], 
                              top_k: int = 10) -> List[Tuple[InformationItem, RelevanceScore]]:
        """Get the top K most relevant items."""
        scored_items = self.batch_score_items(items, user_context)
        return scored_items[:top_k]
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get statistics about scoring performance."""
        if not self.scoring_history:
            return {'message': 'No scoring history available'}
        
        scores = [score.overall_score for _, score, _ in self.scoring_history]
        confidences = [confidence for _, _, confidence in self.scoring_history]
        
        return {
            'total_scored_items': len(self.scoring_history),
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'average_confidence': np.mean(confidences),
            'current_weights': {
                'temporal': self.weights.temporal_weight,
                'personal': self.weights.personal_weight,
                'contextual': self.weights.contextual_weight,
                'pattern': self.weights.pattern_weight,
                'quality': self.weights.quality_weight
            },
            'feedback_count': len(self.user_feedback_history)
        }
