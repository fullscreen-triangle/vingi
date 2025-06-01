#!/usr/bin/env python3
"""
Vingi Cognitive Pattern Recognition Demo

This script demonstrates the core capabilities of the Vingi framework
for detecting and addressing cognitive inefficiency patterns.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the Python path
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
    TaskComplexity,
    NodeType,
    RelationshipType
)


def demo_analysis_paralysis():
    """Demonstrate analysis paralysis detection."""
    print("🔍 Analysis Paralysis Detection Demo")
    print("=" * 50)
    
    # Initialize pattern recognizer
    recognizer = TemporalPatternRecognizer()
    
    # Simulate analysis paralysis behavior
    now = datetime.now()
    
    # Multiple research sessions for the same decision
    behaviors = [
        BehaviorData(
            timestamp=now - timedelta(hours=2),
            action_type="research",
            duration=timedelta(minutes=45),
            decision_count=1,
            task_complexity="medium",
            domain="transportation",
            metadata={"task_id": "train_booking", "information_source": "bahn.de"}
        ),
        BehaviorData(
            timestamp=now - timedelta(hours=1, minutes=30),
            action_type="research", 
            duration=timedelta(minutes=30),
            decision_count=2,
            task_complexity="medium",
            domain="transportation",
            metadata={"task_id": "train_booking", "information_source": "google"}
        ),
        BehaviorData(
            timestamp=now - timedelta(hours=1),
            action_type="comparison",
            duration=timedelta(minutes=25),
            decision_count=3,
            task_complexity="medium", 
            domain="transportation",
            metadata={"task_id": "train_booking", "information_source": "trainline"}
        ),
        BehaviorData(
            timestamp=now - timedelta(minutes=45),
            action_type="research",
            duration=timedelta(minutes=20),
            decision_count=4,
            task_complexity="medium",
            domain="transportation", 
            metadata={"task_id": "train_booking", "information_source": "omio"}
        ),
        BehaviorData(
            timestamp=now - timedelta(minutes=30),
            action_type="comparison",
            duration=timedelta(minutes=35),
            decision_count=5,
            task_complexity="medium",
            domain="transportation",
            metadata={"task_id": "train_booking", "information_source": "reddit"}
        )
    ]
    
    # Add behaviors and check for patterns
    detected_pattern = None
    for behavior in behaviors:
        pattern = recognizer.add_behavior_event(behavior)
        if pattern:
            detected_pattern = pattern
    
    if detected_pattern and detected_pattern.pattern_type == CognitivePatternType.ANALYSIS_PARALYSIS:
        print(f"✅ Analysis Paralysis detected!")
        print(f"   Confidence: {detected_pattern.confidence:.2f}")
        print(f"   Severity: {detected_pattern.severity}")
        print("   Interventions suggested:")
        for intervention in detected_pattern.intervention_suggestions:
            print(f"   • {intervention}")
        print(f"   Context: {detected_pattern.context}")
    else:
        print("❌ No analysis paralysis pattern detected")
    
    print()


def demo_tunnel_vision():
    """Demonstrate tunnel vision planning detection."""
    print("🎯 Tunnel Vision Planning Detection Demo")
    print("=" * 50)
    
    recognizer = TemporalPatternRecognizer()
    now = datetime.now()
    
    # Simulate tunnel vision - only planning transportation, ignoring food
    behaviors = [
        BehaviorData(
            timestamp=now - timedelta(hours=3),
            action_type="planning",
            duration=timedelta(minutes=30),
            domain="transportation",
            metadata={"planning_focus": "train_times"}
        ),
        BehaviorData(
            timestamp=now - timedelta(hours=2, minutes=30),
            action_type="planning",
            duration=timedelta(minutes=25),
            domain="transportation", 
            metadata={"planning_focus": "route_optimization"}
        ),
        BehaviorData(
            timestamp=now - timedelta(hours=2),
            action_type="planning",
            duration=timedelta(minutes=20),
            domain="activities",
            metadata={"planning_focus": "sightseeing"}
        ),
        BehaviorData(
            timestamp=now - timedelta(hours=1, minutes=30),
            action_type="planning",
            duration=timedelta(minutes=40),
            domain="transportation",
            metadata={"planning_focus": "backup_routes"}
        ),
        BehaviorData(
            timestamp=now - timedelta(hours=1),
            action_type="planning", 
            duration=timedelta(minutes=15),
            domain="transportation",
            metadata={"planning_focus": "ticket_booking"}
        )
    ]
    
    detected_pattern = None
    for behavior in behaviors:
        pattern = recognizer.add_behavior_event(behavior)
        if pattern:
            detected_pattern = pattern
    
    if detected_pattern and detected_pattern.pattern_type == CognitivePatternType.TUNNEL_VISION:
        print(f"✅ Tunnel Vision detected!")
        print(f"   Confidence: {detected_pattern.confidence:.2f}")
        print(f"   Severity: {detected_pattern.severity}")
        print("   Interventions suggested:")
        for intervention in detected_pattern.intervention_suggestions:
            print(f"   • {intervention}")
        print(f"   Missing domains: {detected_pattern.context.get('missing_critical_domains', [])}")
    else:
        print("❌ No tunnel vision pattern detected")
    
    print()


def demo_default_behavior_loops():
    """Demonstrate default behavior loop detection."""
    print("🔄 Default Behavior Loop Detection Demo")
    print("=" * 50)
    
    recognizer = TemporalPatternRecognizer()
    now = datetime.now()
    
    # Simulate repetitive choices (always Five Guys)
    behaviors = []
    for i in range(15):
        behaviors.append(BehaviorData(
            timestamp=now - timedelta(days=i, hours=12),
            action_type="choice",
            domain="food",
            metadata={"choice_made": "Five Guys burger"}
        ))
    
    # Add a few other repetitive patterns
    for i in range(10):
        behaviors.append(BehaviorData(
            timestamp=now - timedelta(days=i, hours=18),
            action_type="purchase",
            domain="shopping", 
            metadata={"choice_made": "nearby_supermarket_bread"}
        ))
    
    detected_pattern = None
    for behavior in behaviors:
        pattern = recognizer.add_behavior_event(behavior)
        if pattern:
            detected_pattern = pattern
    
    if detected_pattern and detected_pattern.pattern_type == CognitivePatternType.DEFAULT_BEHAVIOR_LOOP:
        print(f"✅ Default Behavior Loop detected!")
        print(f"   Confidence: {detected_pattern.confidence:.2f}")
        print(f"   Severity: {detected_pattern.severity}")
        print(f"   Most repeated choice: {detected_pattern.context.get('most_repeated_choice')}")
        print(f"   Repetition rate: {detected_pattern.context.get('repetition_rate'):.2f}")
        print("   Interventions suggested:")
        for intervention in detected_pattern.intervention_suggestions:
            print(f"   • {intervention}")
    else:
        print("❌ No default behavior loop detected")
    
    print()


def demo_context_graph():
    """Demonstrate context graph functionality."""
    print("🕸️  Context Graph Management Demo")
    print("=" * 50)
    
    # Initialize context manager
    context_manager = ContextGraphManager()
    
    # Add some preferences
    context_manager.record_preference(
        item="fresh_bread",
        preference_type="like",
        strength=0.9,
        context={"source": "bakery", "quality": "artisanal"}
    )
    
    context_manager.record_preference(
        item="store_brand_bread", 
        preference_type="dislike",
        strength=0.7,
        context={"source": "supermarket", "quality": "processed"}
    )
    
    # Add activity context
    context_manager.add_activity_context(
        activity_name="train_research",
        domain="transportation",
        properties={"complexity": "medium", "frequency": "occasional"}
    )
    
    # Get preferences
    food_preferences = context_manager.get_user_preferences("food")
    print("Food preferences:")
    print(f"   Likes: {len(food_preferences['likes'])} items")
    for like in food_preferences['likes']:
        print(f"     • {like['item']} (strength: {like['strength']})")
    print(f"   Dislikes: {len(food_preferences['dislikes'])} items") 
    for dislike in food_preferences['dislikes']:
        print(f"     • {dislike['item']} (strength: {dislike['strength']})")
    
    # Analyze patterns
    analysis = context_manager.analyze_patterns()
    print(f"\nGraph analysis:")
    print(f"   Total nodes: {analysis['node_distribution']}")
    print(f"   Relationships: {analysis['relationship_patterns']['total_relationships']}")
    
    print()


def demo_relevance_scoring():
    """Demonstrate relevance scoring."""
    print("⭐ Relevance Scoring Demo")
    print("=" * 50)
    
    # Initialize components
    context_manager = ContextGraphManager()
    pattern_recognizer = TemporalPatternRecognizer()
    relevance_scorer = RelevanceScorer(context_manager, pattern_recognizer)
    
    # Create some information items to score
    items = [
        InformationItem(
            title="Quick Train Booking Guide",
            content="Simple 5-step process to book train tickets without analysis paralysis",
            source="travel_blog.com",
            domain="transportation",
            tags=["quick", "simple", "trains"],
            timestamp=datetime.now()
        ),
        InformationItem(
            title="Comprehensive Train Route Analysis",
            content="Detailed comparison of 47 different route options with price analysis",
            source="transport_research.edu",
            domain="transportation", 
            tags=["detailed", "analysis", "comprehensive"],
            timestamp=datetime.now()
        ),
        InformationItem(
            title="Best Bakeries in Nuremberg",
            content="Local guide to artisanal bread shops and fresh pastries",
            source="local_food_guide.de",
            domain="food",
            tags=["local", "bakery", "fresh"],
            timestamp=datetime.now()
        )
    ]
    
    # Create user context suggesting analysis paralysis
    user_context = {
        "current_task": {
            "keywords": ["train", "booking", "quick"]
        },
        "time_context": {
            "urgency_level": "high"
        },
        "active_domains": ["transportation"]
    }
    
    # Score relevance
    scored_items = relevance_scorer.batch_score_items(items, user_context)
    
    print("Relevance scores (for user experiencing analysis paralysis):")
    for item, score in scored_items:
        print(f"\n   📄 {item.title}")
        print(f"      Overall score: {score.overall_score:.2f}")
        print(f"      Temporal: {score.temporal_score:.2f} | Personal: {score.personal_score:.2f}")
        print(f"      Contextual: {score.contextual_score:.2f} | Pattern: {score.pattern_score:.2f}")
        print(f"      Quality: {score.quality_score:.2f} | Confidence: {score.confidence:.2f}")
        if score.reasoning:
            print(f"      Reasoning: {', '.join(score.reasoning)}")
    
    print()


def demo_temporal_analysis():
    """Demonstrate temporal analysis."""
    print("⏰ Temporal Analysis Demo")
    print("=" * 50)
    
    analyzer = TemporalAnalyzer()
    now = datetime.now()
    
    # Add some temporal events simulating work patterns
    events = [
        # Morning high-energy work
        TemporalEvent(
            timestamp=now.replace(hour=9, minute=0) - timedelta(days=1),
            event_type="coding",
            duration=timedelta(hours=2),
            energy_level=0.9,
            focus_level=0.8,
            task_complexity=TaskComplexity.HIGH,
            completion_status=True,
            interruptions=1
        ),
        TemporalEvent(
            timestamp=now.replace(hour=10, minute=30) - timedelta(days=1),
            event_type="coding",
            duration=timedelta(hours=1, minutes=30),
            energy_level=0.8,
            focus_level=0.9,
            task_complexity=TaskComplexity.HIGH,
            completion_status=True,
            interruptions=0
        ),
        # Post-lunch lower energy
        TemporalEvent(
            timestamp=now.replace(hour=13, minute=30) - timedelta(days=1),
            event_type="emails",
            duration=timedelta(minutes=45),
            energy_level=0.4,
            focus_level=0.3,
            task_complexity=TaskComplexity.LOW,
            completion_status=True,
            interruptions=3
        ),
        # Afternoon recovery
        TemporalEvent(
            timestamp=now.replace(hour=15, minute=0) - timedelta(days=1),
            event_type="writing",
            duration=timedelta(hours=1),
            energy_level=0.7,
            focus_level=0.6,
            task_complexity=TaskComplexity.MEDIUM,
            completion_status=True,
            interruptions=1
        )
    ]
    
    # Add events to analyzer
    for event in events:
        analyzer.add_event(event)
    
    # Get optimal time slots for coding
    optimal_slots = analyzer.get_optimal_time_slots(
        task_type="coding",
        duration=timedelta(hours=2),
        days_ahead=3
    )
    
    print("Optimal time slots for 2-hour coding sessions:")
    for i, slot in enumerate(optimal_slots[:3]):
        print(f"\n   {i+1}. {slot.start_time} - {slot.end_time}")
        print(f"      Day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][slot.day_of_week]}")
        print(f"      Energy prediction: {slot.energy_prediction:.2f}")
        print(f"      Focus prediction: {slot.focus_prediction:.2f}")
        print(f"      Confidence: {slot.confidence:.2f}")
        if slot.reasoning:
            print(f"      Reasoning: {', '.join(slot.reasoning)}")
    
    # Predict energy levels for tomorrow
    tomorrow = datetime.now() + timedelta(days=1)
    energy_predictions = analyzer.predict_energy_levels(tomorrow)
    
    print(f"\nEnergy predictions for tomorrow:")
    peak_hours = sorted(energy_predictions.items(), key=lambda x: x[1], reverse=True)[:3]
    print("   Top 3 energy hours:")
    for hour, energy in peak_hours:
        print(f"     {hour:02d}:00 - Energy level: {energy:.2f}")
    
    print()


def main():
    """Run all demos."""
    print("🧠 Vingi Cognitive Load Optimization Framework Demo")
    print("=" * 60)
    print()
    
    try:
        demo_analysis_paralysis()
        demo_tunnel_vision() 
        demo_default_behavior_loops()
        demo_context_graph()
        demo_relevance_scoring()
        demo_temporal_analysis()
        
        print("✅ All demos completed successfully!")
        print("\nNext steps:")
        print("• Integrate with your calendar and email")
        print("• Add real behavioral data collection")
        print("• Set up automated interventions")
        print("• Customize pattern thresholds for your needs")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 