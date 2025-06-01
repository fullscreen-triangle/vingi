"""
Temporal Analysis for Personal Activity Optimization

This module analyzes temporal patterns in user behavior to optimize task scheduling,
predict optimal work periods, and identify time-based cognitive patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import logging
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TemporalPatternType(Enum):
    """Types of temporal patterns."""
    DAILY_RHYTHM = "daily_rhythm"
    WEEKLY_CYCLE = "weekly_cycle"
    SEASONAL_TREND = "seasonal_trend"
    ENERGY_CYCLE = "energy_cycle"
    FOCUS_PERIOD = "focus_period"
    INTERRUPTION_PATTERN = "interruption_pattern"
    DEADLINE_BEHAVIOR = "deadline_behavior"


class TaskComplexity(Enum):
    """Task complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


@dataclass
class TemporalEvent:
    """Represents a timestamped event for analysis."""
    timestamp: datetime
    event_type: str
    duration: Optional[timedelta] = None
    energy_level: Optional[float] = None  # 0.0 to 1.0
    focus_level: Optional[float] = None   # 0.0 to 1.0
    task_complexity: Optional[TaskComplexity] = None
    completion_status: Optional[bool] = None
    interruptions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalPattern:
    """Represents a discovered temporal pattern."""
    pattern_type: TemporalPatternType
    description: str
    confidence: float  # 0.0 to 1.0
    time_range: Tuple[time, time]  # Start and end times
    days_of_week: List[int] = field(default_factory=list)  # 0=Monday, 6=Sunday
    peak_value: float = 0.0
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimalTimeSlot:
    """Represents an optimal time slot for specific activities."""
    start_time: time
    end_time: time
    day_of_week: Optional[int] = None
    task_types: List[str] = field(default_factory=list)
    energy_prediction: float = 0.5
    focus_prediction: float = 0.5
    confidence: float = 0.5
    reasoning: List[str] = field(default_factory=list)


class TemporalAnalyzer:
    """
    Analyzes temporal patterns in user behavior to optimize scheduling
    and predict optimal work periods.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the temporal analyzer.
        
        Args:
            storage_path: Path for storing temporal data
        """
        self.storage_path = storage_path or Path.home() / "Library" / "Application Support" / "Vingi" / "temporal_data.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Event storage
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.patterns: List[TemporalPattern] = []
        
        # Analysis parameters
        self.min_pattern_confidence = 0.6
        self.analysis_window_days = 30
        self.energy_smoothing_window = 5
        
        # Load existing data
        self._load_data()
    
    def add_event(self, event: TemporalEvent):
        """
        Add a temporal event for analysis.
        
        Args:
            event: The temporal event to add
        """
        self.events.append(event)
        logger.debug(f"Added temporal event: {event.event_type} at {event.timestamp}")
        
        # Trigger pattern analysis if we have enough recent data
        if len(self.events) % 50 == 0:  # Analyze every 50 events
            self._analyze_patterns()
    
    def get_optimal_time_slots(self, task_type: str, 
                              duration: timedelta,
                              days_ahead: int = 7) -> List[OptimalTimeSlot]:
        """
        Get optimal time slots for a specific task type.
        
        Args:
            task_type: Type of task to schedule
            duration: Required duration for the task
            days_ahead: Number of days ahead to consider
            
        Returns:
            List of optimal time slots sorted by predicted effectiveness
        """
        optimal_slots = []
        
        # Analyze historical performance for this task type
        task_events = [e for e in self.events if e.event_type == task_type]
        
        if not task_events:
            # No historical data, return general optimal slots
            return self._get_general_optimal_slots(duration, days_ahead)
        
        # Calculate performance metrics by time of day
        hourly_performance = defaultdict(list)
        for event in task_events:
            hour = event.timestamp.hour
            performance = self._calculate_event_performance(event)
            hourly_performance[hour].append(performance)
        
        # Find optimal hours
        hour_scores = {}
        for hour, performances in hourly_performance.items():
            if len(performances) >= 3:  # Need at least 3 data points
                avg_performance = np.mean(performances)
                consistency = 1.0 - np.std(performances)  # Lower std = more consistent
                hour_scores[hour] = avg_performance * 0.7 + consistency * 0.3
        
        # Generate time slots for upcoming days
        now = datetime.now()
        for day_offset in range(days_ahead):
            target_date = now.date() + timedelta(days=day_offset)
            day_of_week = target_date.weekday()
            
            # Check weekly patterns
            weekly_multiplier = self._get_weekly_multiplier(day_of_week, task_type)
            
            for hour, score in hour_scores.items():
                if score >= 0.6:  # Only consider good time slots
                    start_time = time(hour, 0)
                    end_time = self._add_duration_to_time(start_time, duration)
                    
                    if end_time is None:  # Duration extends past midnight
                        continue
                    
                    # Calculate predictions
                    energy_pred = self._predict_energy_level(day_of_week, hour)
                    focus_pred = self._predict_focus_level(day_of_week, hour, task_type)
                    
                    # Adjust for weekly patterns
                    adjusted_score = score * weekly_multiplier
                    
                    slot = OptimalTimeSlot(
                        start_time=start_time,
                        end_time=end_time,
                        day_of_week=day_of_week,
                        task_types=[task_type],
                        energy_prediction=energy_pred,
                        focus_prediction=focus_pred,
                        confidence=min(adjusted_score, 1.0),
                        reasoning=self._generate_slot_reasoning(hour, score, weekly_multiplier)
                    )
                    optimal_slots.append(slot)
        
        # Sort by confidence and return top slots
        optimal_slots.sort(key=lambda s: s.confidence, reverse=True)
        return optimal_slots[:10]  # Return top 10 slots
    
    def predict_energy_levels(self, target_date: datetime) -> Dict[int, float]:
        """
        Predict energy levels for each hour of a specific day.
        
        Args:
            target_date: Date to predict energy levels for
            
        Returns:
            Dictionary mapping hour (0-23) to predicted energy level (0.0-1.0)
        """
        day_of_week = target_date.weekday()
        predictions = {}
        
        # Get historical energy data for this day of week
        same_day_events = [e for e in self.events 
                          if e.timestamp.weekday() == day_of_week 
                          and e.energy_level is not None]
        
        if not same_day_events:
            # No data for this day, use general pattern
            return self._get_default_energy_pattern()
        
        # Calculate average energy by hour for this day of week
        hourly_energy = defaultdict(list)
        for event in same_day_events:
            hour = event.timestamp.hour
            hourly_energy[hour].append(event.energy_level)
        
        # Generate predictions with smoothing
        base_predictions = {}
        for hour in range(24):
            if hour in hourly_energy:
                base_predictions[hour] = np.mean(hourly_energy[hour])
            else:
                # Interpolate from nearby hours
                base_predictions[hour] = self._interpolate_energy(hour, hourly_energy)
        
        # Apply smoothing
        for hour in range(24):
            smoothed_energy = self._smooth_energy_prediction(hour, base_predictions)
            predictions[hour] = max(0.0, min(1.0, smoothed_energy))
        
        return predictions
    
    def analyze_productivity_patterns(self) -> Dict[str, Any]:
        """
        Analyze overall productivity patterns.
        
        Returns:
            Dictionary containing productivity analysis results
        """
        if len(self.events) < 50:
            return {'error': 'Insufficient data for analysis'}
        
        analysis = {
            'daily_patterns': self._analyze_daily_patterns(),
            'weekly_patterns': self._analyze_weekly_patterns(),
            'task_performance': self._analyze_task_performance(),
            'energy_trends': self._analyze_energy_trends(),
            'focus_patterns': self._analyze_focus_patterns(),
            'optimal_work_blocks': self._identify_optimal_work_blocks()
        }
        
        return analysis
    
    def _analyze_patterns(self):
        """Analyze temporal patterns in the event data."""
        self.patterns.clear()
        
        # Analyze different pattern types
        daily_patterns = self._detect_daily_rhythms()
        weekly_patterns = self._detect_weekly_cycles()
        energy_patterns = self._detect_energy_cycles()
        focus_patterns = self._detect_focus_periods()
        
        # Combine all patterns
        self.patterns.extend(daily_patterns)
        self.patterns.extend(weekly_patterns)
        self.patterns.extend(energy_patterns)
        self.patterns.extend(focus_patterns)
        
        # Filter by confidence
        self.patterns = [p for p in self.patterns if p.confidence >= self.min_pattern_confidence]
        
        logger.info(f"Detected {len(self.patterns)} temporal patterns")
    
    def _detect_daily_rhythms(self) -> List[TemporalPattern]:
        """Detect daily rhythm patterns."""
        patterns = []
        
        # Group events by hour of day
        hourly_events = defaultdict(list)
        for event in self.events:
            hour = event.timestamp.hour
            hourly_events[hour].append(event)
        
        # Find peak activity hours
        hourly_activity = {}
        for hour, events in hourly_events.items():
            if len(events) >= 5:  # Need sufficient data
                avg_energy = np.mean([e.energy_level for e in events if e.energy_level is not None])
                avg_focus = np.mean([e.focus_level for e in events if e.focus_level is not None])
                hourly_activity[hour] = (avg_energy or 0.5) * 0.5 + (avg_focus or 0.5) * 0.5
        
        if hourly_activity:
            peak_hour = max(hourly_activity.items(), key=lambda x: x[1])
            if peak_hour[1] > 0.7:  # High activity threshold
                pattern = TemporalPattern(
                    pattern_type=TemporalPatternType.DAILY_RHYTHM,
                    description=f"Peak productivity at {peak_hour[0]}:00",
                    confidence=min(peak_hour[1], 1.0),
                    time_range=(time(peak_hour[0], 0), time(peak_hour[0] + 1, 0)),
                    peak_value=peak_hour[1],
                    metadata={'hourly_activity': dict(hourly_activity)}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_weekly_cycles(self) -> List[TemporalPattern]:
        """Detect weekly cycle patterns."""
        patterns = []
        
        # Group events by day of week
        daily_events = defaultdict(list)
        for event in self.events:
            day_of_week = event.timestamp.weekday()
            daily_events[day_of_week].append(event)
        
        # Calculate daily productivity scores
        daily_productivity = {}
        for day, events in daily_events.items():
            if len(events) >= 10:  # Need sufficient data
                productivity_scores = []
                for event in events:
                    score = self._calculate_event_performance(event)
                    productivity_scores.append(score)
                
                if productivity_scores:
                    daily_productivity[day] = np.mean(productivity_scores)
        
        if len(daily_productivity) >= 5:  # Need data for most days
            # Find most and least productive days
            max_day = max(daily_productivity.items(), key=lambda x: x[1])
            min_day = min(daily_productivity.items(), key=lambda x: x[1])
            
            if max_day[1] - min_day[1] > 0.3:  # Significant difference
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                pattern = TemporalPattern(
                    pattern_type=TemporalPatternType.WEEKLY_CYCLE,
                    description=f"Best day: {day_names[max_day[0]]}, Worst day: {day_names[min_day[0]]}",
                    confidence=max_day[1] - min_day[1],
                    time_range=(time(0, 0), time(23, 59)),
                    days_of_week=list(daily_productivity.keys()),
                    peak_value=max_day[1],
                    metadata={'daily_productivity': daily_productivity}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_energy_cycles(self) -> List[TemporalPattern]:
        """Detect energy cycle patterns."""
        patterns = []
        
        energy_events = [e for e in self.events if e.energy_level is not None]
        if len(energy_events) < 20:
            return patterns
        
        # Group by hour and calculate average energy
        hourly_energy = defaultdict(list)
        for event in energy_events:
            hour = event.timestamp.hour
            hourly_energy[hour].append(event.energy_level)
        
        avg_hourly_energy = {}
        for hour, energies in hourly_energy.items():
            if len(energies) >= 3:
                avg_hourly_energy[hour] = np.mean(energies)
        
        if len(avg_hourly_energy) >= 8:  # Need good coverage
            # Find energy peaks and valleys
            sorted_hours = sorted(avg_hourly_energy.items())
            
            # Simple peak detection
            peaks = []
            valleys = []
            
            for i in range(1, len(sorted_hours) - 1):
                prev_energy = sorted_hours[i-1][1]
                curr_energy = sorted_hours[i][1]
                next_energy = sorted_hours[i+1][1]
                
                if curr_energy > prev_energy and curr_energy > next_energy and curr_energy > 0.7:
                    peaks.append(sorted_hours[i])
                elif curr_energy < prev_energy and curr_energy < next_energy and curr_energy < 0.4:
                    valleys.append(sorted_hours[i])
            
            if peaks:
                main_peak = max(peaks, key=lambda x: x[1])
                pattern = TemporalPattern(
                    pattern_type=TemporalPatternType.ENERGY_CYCLE,
                    description=f"Energy peak at {main_peak[0]}:00",
                    confidence=main_peak[1],
                    time_range=(time(main_peak[0], 0), time(main_peak[0] + 1, 0)),
                    peak_value=main_peak[1],
                    metadata={
                        'all_peaks': peaks,
                        'valleys': valleys,
                        'hourly_energy': avg_hourly_energy
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_focus_periods(self) -> List[TemporalPattern]:
        """Detect focus period patterns."""
        patterns = []
        
        focus_events = [e for e in self.events if e.focus_level is not None]
        if len(focus_events) < 20:
            return patterns
        
        # Find periods of sustained high focus
        focus_sessions = []
        current_session = []
        
        for event in sorted(focus_events, key=lambda e: e.timestamp):
            if event.focus_level >= 0.7:  # High focus threshold
                current_session.append(event)
            else:
                if len(current_session) >= 2:  # Minimum session length
                    focus_sessions.append(current_session)
                current_session = []
        
        # Add the last session if it qualifies
        if len(current_session) >= 2:
            focus_sessions.append(current_session)
        
        if focus_sessions:
            # Find the most common focus period time
            session_hours = []
            for session in focus_sessions:
                start_hour = session[0].timestamp.hour
                session_hours.append(start_hour)
            
            if session_hours:
                # Find most common start hour
                hour_counts = defaultdict(int)
                for hour in session_hours:
                    hour_counts[hour] += 1
                
                most_common_hour = max(hour_counts.items(), key=lambda x: x[1])
                if most_common_hour[1] >= 3:  # At least 3 occurrences
                    avg_focus = np.mean([s[0].focus_level for s in focus_sessions 
                                       if s[0].timestamp.hour == most_common_hour[0]])
                    
                    pattern = TemporalPattern(
                        pattern_type=TemporalPatternType.FOCUS_PERIOD,
                        description=f"High focus period starting at {most_common_hour[0]}:00",
                        confidence=avg_focus,
                        time_range=(time(most_common_hour[0], 0), time(most_common_hour[0] + 2, 0)),
                        peak_value=avg_focus,
                        metadata={
                            'session_count': len(focus_sessions),
                            'hour_distribution': dict(hour_counts)
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_event_performance(self, event: TemporalEvent) -> float:
        """Calculate performance score for an event."""
        score = 0.5  # Base score
        
        # Factor in energy level
        if event.energy_level is not None:
            score += (event.energy_level - 0.5) * 0.3
        
        # Factor in focus level
        if event.focus_level is not None:
            score += (event.focus_level - 0.5) * 0.3
        
        # Factor in completion status
        if event.completion_status is not None:
            score += 0.2 if event.completion_status else -0.2
        
        # Factor in interruptions (fewer is better)
        interruption_penalty = min(0.2, event.interruptions * 0.05)
        score -= interruption_penalty
        
        # Factor in task complexity vs energy (harder tasks need more energy)
        if event.task_complexity and event.energy_level:
            complexity_energy_match = self._calculate_complexity_energy_match(
                event.task_complexity, event.energy_level
            )
            score += complexity_energy_match * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_complexity_energy_match(self, complexity: TaskComplexity, energy: float) -> float:
        """Calculate how well task complexity matches energy level."""
        complexity_requirements = {
            TaskComplexity.LOW: 0.3,
            TaskComplexity.MEDIUM: 0.5,
            TaskComplexity.HIGH: 0.8,
            TaskComplexity.CREATIVE: 0.7,
            TaskComplexity.ANALYTICAL: 0.8
        }
        
        required_energy = complexity_requirements.get(complexity, 0.5)
        # Good match when energy >= required, penalty when energy < required
        if energy >= required_energy:
            return min(1.0, energy / required_energy - 1.0)
        else:
            return (energy / required_energy) - 1.0
    
    def _get_general_optimal_slots(self, duration: timedelta, days_ahead: int) -> List[OptimalTimeSlot]:
        """Get general optimal slots when no task-specific data available."""
        slots = []
        
        # General optimal work hours based on research
        optimal_hours = [9, 10, 11, 14, 15, 16]  # 9-11 AM and 2-4 PM
        
        now = datetime.now()
        for day_offset in range(days_ahead):
            target_date = now.date() + timedelta(days=day_offset)
            day_of_week = target_date.weekday()
            
            # Skip weekends for work tasks
            if day_of_week >= 5:  # Saturday or Sunday
                continue
            
            for hour in optimal_hours:
                start_time = time(hour, 0)
                end_time = self._add_duration_to_time(start_time, duration)
                
                if end_time is None:
                    continue
                
                slot = OptimalTimeSlot(
                    start_time=start_time,
                    end_time=end_time,
                    day_of_week=day_of_week,
                    task_types=[],
                    energy_prediction=0.7,
                    focus_prediction=0.7,
                    confidence=0.6,
                    reasoning=["General optimal work hours", "No task-specific data available"]
                )
                slots.append(slot)
        
        return slots
    
    def _add_duration_to_time(self, start_time: time, duration: timedelta) -> Optional[time]:
        """Add duration to time, return None if it goes past midnight."""
        start_datetime = datetime.combine(datetime.today(), start_time)
        end_datetime = start_datetime + duration
        
        if end_datetime.date() > start_datetime.date():
            return None  # Spans multiple days
        
        return end_datetime.time()
    
    def _get_weekly_multiplier(self, day_of_week: int, task_type: str) -> float:
        """Get weekly pattern multiplier for a specific day and task type."""
        # Find weekly patterns
        weekly_patterns = [p for p in self.patterns 
                          if p.pattern_type == TemporalPatternType.WEEKLY_CYCLE]
        
        if not weekly_patterns:
            return 1.0  # No pattern data
        
        pattern = weekly_patterns[0]  # Use the first/strongest pattern
        daily_productivity = pattern.metadata.get('daily_productivity', {})
        
        if day_of_week in daily_productivity:
            # Normalize around 1.0
            avg_productivity = np.mean(list(daily_productivity.values()))
            day_productivity = daily_productivity[day_of_week]
            return day_productivity / avg_productivity if avg_productivity > 0 else 1.0
        
        return 1.0
    
    def _predict_energy_level(self, day_of_week: int, hour: int) -> float:
        """Predict energy level for a specific day and hour."""
        # Find energy patterns
        energy_patterns = [p for p in self.patterns 
                          if p.pattern_type == TemporalPatternType.ENERGY_CYCLE]
        
        if not energy_patterns:
            return self._get_default_energy_for_hour(hour)
        
        pattern = energy_patterns[0]
        hourly_energy = pattern.metadata.get('hourly_energy', {})
        
        if hour in hourly_energy:
            return hourly_energy[hour]
        
        return self._get_default_energy_for_hour(hour)
    
    def _predict_focus_level(self, day_of_week: int, hour: int, task_type: str) -> float:
        """Predict focus level for a specific day, hour, and task type."""
        # Find focus patterns
        focus_patterns = [p for p in self.patterns 
                         if p.pattern_type == TemporalPatternType.FOCUS_PERIOD]
        
        base_focus = 0.5
        
        for pattern in focus_patterns:
            pattern_start_hour = pattern.time_range[0].hour
            pattern_end_hour = pattern.time_range[1].hour
            
            if pattern_start_hour <= hour <= pattern_end_hour:
                base_focus = max(base_focus, pattern.peak_value)
        
        return base_focus
    
    def _get_default_energy_for_hour(self, hour: int) -> float:
        """Get default energy level for an hour based on circadian rhythm."""
        # Simple circadian rhythm approximation
        if 6 <= hour <= 9:
            return 0.6  # Morning energy ramp-up
        elif 10 <= hour <= 11:
            return 0.8  # Morning peak
        elif 12 <= hour <= 13:
            return 0.5  # Post-lunch dip
        elif 14 <= hour <= 16:
            return 0.7  # Afternoon productivity
        elif 17 <= hour <= 19:
            return 0.6  # Evening
        elif 20 <= hour <= 22:
            return 0.4  # Evening wind-down
        else:
            return 0.2  # Night/early morning
    
    def _get_default_energy_pattern(self) -> Dict[int, float]:
        """Get default 24-hour energy pattern."""
        return {hour: self._get_default_energy_for_hour(hour) for hour in range(24)}
    
    def _interpolate_energy(self, target_hour: int, hourly_energy: Dict[int, List[float]]) -> float:
        """Interpolate energy level for an hour with no data."""
        # Find nearest hours with data
        available_hours = sorted(hourly_energy.keys())
        
        if not available_hours:
            return self._get_default_energy_for_hour(target_hour)
        
        # Find closest hours
        before_hours = [h for h in available_hours if h <= target_hour]
        after_hours = [h for h in available_hours if h > target_hour]
        
        if not before_hours:
            # Use first available hour
            return np.mean(hourly_energy[available_hours[0]])
        
        if not after_hours:
            # Use last available hour
            return np.mean(hourly_energy[available_hours[-1]])
        
        # Interpolate between closest hours
        before_hour = max(before_hours)
        after_hour = min(after_hours)
        
        before_energy = np.mean(hourly_energy[before_hour])
        after_energy = np.mean(hourly_energy[after_hour])
        
        # Linear interpolation
        weight = (target_hour - before_hour) / (after_hour - before_hour)
        return before_energy * (1 - weight) + after_energy * weight
    
    def _smooth_energy_prediction(self, hour: int, predictions: Dict[int, float]) -> float:
        """Apply smoothing to energy predictions."""
        # Simple moving average smoothing
        window_size = self.energy_smoothing_window
        hours_to_average = []
        
        for offset in range(-window_size//2, window_size//2 + 1):
            target_hour = (hour + offset) % 24
            if target_hour in predictions:
                hours_to_average.append(predictions[target_hour])
        
        if hours_to_average:
            return np.mean(hours_to_average)
        else:
            return predictions.get(hour, 0.5)
    
    def _generate_slot_reasoning(self, hour: int, base_score: float, weekly_multiplier: float) -> List[str]:
        """Generate reasoning for a time slot recommendation."""
        reasoning = []
        
        if base_score > 0.8:
            reasoning.append(f"Historically high performance at {hour}:00")
        elif base_score > 0.6:
            reasoning.append(f"Good historical performance at {hour}:00")
        
        if weekly_multiplier > 1.1:
            reasoning.append("Strong day-of-week effect")
        elif weekly_multiplier < 0.9:
            reasoning.append("Weaker day-of-week effect")
        
        # Add energy-related reasoning
        if 9 <= hour <= 11:
            reasoning.append("Morning energy peak period")
        elif 14 <= hour <= 16:
            reasoning.append("Afternoon productivity window")
        
        return reasoning
    
    def _analyze_daily_patterns(self) -> Dict[str, Any]:
        """Analyze daily patterns in detail."""
        hourly_stats = defaultdict(lambda: {'events': 0, 'avg_energy': 0, 'avg_focus': 0, 'performance': []})
        
        for event in self.events:
            hour = event.timestamp.hour
            hourly_stats[hour]['events'] += 1
            
            if event.energy_level is not None:
                hourly_stats[hour]['avg_energy'] += event.energy_level
            if event.focus_level is not None:
                hourly_stats[hour]['avg_focus'] += event.focus_level
            
            performance = self._calculate_event_performance(event)
            hourly_stats[hour]['performance'].append(performance)
        
        # Calculate averages
        for hour_data in hourly_stats.values():
            if hour_data['events'] > 0:
                hour_data['avg_energy'] /= hour_data['events']
                hour_data['avg_focus'] /= hour_data['events']
                hour_data['avg_performance'] = np.mean(hour_data['performance']) if hour_data['performance'] else 0
        
        return dict(hourly_stats)
    
    def _analyze_weekly_patterns(self) -> Dict[str, Any]:
        """Analyze weekly patterns in detail."""
        daily_stats = defaultdict(lambda: {'events': 0, 'avg_performance': 0, 'total_duration': timedelta()})
        
        for event in self.events:
            day_of_week = event.timestamp.weekday()
            daily_stats[day_of_week]['events'] += 1
            
            performance = self._calculate_event_performance(event)
            daily_stats[day_of_week]['avg_performance'] += performance
            
            if event.duration:
                daily_stats[day_of_week]['total_duration'] += event.duration
        
        # Calculate averages
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        result = {}
        
        for day_num, day_data in daily_stats.items():
            if day_data['events'] > 0:
                result[day_names[day_num]] = {
                    'events': day_data['events'],
                    'avg_performance': day_data['avg_performance'] / day_data['events'],
                    'total_hours': day_data['total_duration'].total_seconds() / 3600
                }
        
        return result
    
    def _analyze_task_performance(self) -> Dict[str, Any]:
        """Analyze task performance patterns."""
        task_stats = defaultdict(lambda: {'count': 0, 'performance': [], 'completion_rate': 0, 'completed': 0})
        
        for event in self.events:
            task_type = event.event_type
            performance = self._calculate_event_performance(event)
            
            task_stats[task_type]['count'] += 1
            task_stats[task_type]['performance'].append(performance)
            
            if event.completion_status is not None:
                if event.completion_status:
                    task_stats[task_type]['completed'] += 1
        
        # Calculate final metrics
        result = {}
        for task_type, stats in task_stats.items():
            if stats['count'] >= 3:  # Need minimum data
                result[task_type] = {
                    'total_attempts': stats['count'],
                    'avg_performance': np.mean(stats['performance']),
                    'performance_consistency': 1.0 - np.std(stats['performance']),
                    'completion_rate': stats['completed'] / stats['count'] if stats['count'] > 0 else 0
                }
        
        return result
    
    def _analyze_energy_trends(self) -> Dict[str, Any]:
        """Analyze energy trends over time."""
        if not any(e.energy_level is not None for e in self.events):
            return {'error': 'No energy data available'}
        
        energy_events = [e for e in self.events if e.energy_level is not None]
        energy_events.sort(key=lambda e: e.timestamp)
        
        # Calculate weekly averages
        weekly_averages = []
        current_week_events = []
        current_week_start = None
        
        for event in energy_events:
            week_start = event.timestamp - timedelta(days=event.timestamp.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if current_week_start is None:
                current_week_start = week_start
            
            if week_start == current_week_start:
                current_week_events.append(event)
            else:
                # Process completed week
                if current_week_events:
                    avg_energy = np.mean([e.energy_level for e in current_week_events])
                    weekly_averages.append(avg_energy)
                
                current_week_events = [event]
                current_week_start = week_start
        
        # Process last week
        if current_week_events:
            avg_energy = np.mean([e.energy_level for e in current_week_events])
            weekly_averages.append(avg_energy)
        
        # Calculate trend
        trend = "stable"
        if len(weekly_averages) >= 3:
            recent_avg = np.mean(weekly_averages[-2:])
            earlier_avg = np.mean(weekly_averages[:-2])
            
            if recent_avg > earlier_avg + 0.1:
                trend = "increasing"
            elif recent_avg < earlier_avg - 0.1:
                trend = "decreasing"
        
        return {
            'weekly_averages': weekly_averages,
            'current_trend': trend,
            'overall_average': np.mean([e.energy_level for e in energy_events]),
            'energy_stability': 1.0 - np.std([e.energy_level for e in energy_events])
        }
    
    def _analyze_focus_patterns(self) -> Dict[str, Any]:
        """Analyze focus patterns in detail."""
        if not any(e.focus_level is not None for e in self.events):
            return {'error': 'No focus data available'}
        
        focus_events = [e for e in self.events if e.focus_level is not None]
        
        # Analyze focus by task complexity
        complexity_focus = defaultdict(list)
        for event in focus_events:
            if event.task_complexity:
                complexity_focus[event.task_complexity.value].append(event.focus_level)
        
        complexity_analysis = {}
        for complexity, focus_levels in complexity_focus.items():
            if len(focus_levels) >= 3:
                complexity_analysis[complexity] = {
                    'avg_focus': np.mean(focus_levels),
                    'focus_consistency': 1.0 - np.std(focus_levels),
                    'sample_size': len(focus_levels)
                }
        
        # Find focus sessions (periods of sustained high focus)
        focus_sessions = []
        current_session = []
        
        for event in sorted(focus_events, key=lambda e: e.timestamp):
            if event.focus_level >= 0.7:
                current_session.append(event)
            else:
                if len(current_session) >= 2:
                    session_duration = current_session[-1].timestamp - current_session[0].timestamp
                    focus_sessions.append({
                        'start': current_session[0].timestamp,
                        'duration_minutes': session_duration.total_seconds() / 60,
                        'avg_focus': np.mean([e.focus_level for e in current_session])
                    })
                current_session = []
        
        return {
            'overall_focus_average': np.mean([e.focus_level for e in focus_events]),
            'focus_by_complexity': complexity_analysis,
            'focus_sessions': focus_sessions[-10:],  # Last 10 sessions
            'total_focus_sessions': len(focus_sessions)
        }
    
    def _identify_optimal_work_blocks(self) -> List[Dict[str, Any]]:
        """Identify optimal contiguous work blocks."""
        # Group events by day
        daily_events = defaultdict(list)
        for event in self.events:
            date = event.timestamp.date()
            daily_events[date].append(event)
        
        work_blocks = []
        
        for date, events in daily_events.items():
            events.sort(key=lambda e: e.timestamp)
            
            # Find contiguous high-performance periods
            current_block = []
            for event in events:
                performance = self._calculate_event_performance(event)
                
                if performance >= 0.6:  # High performance threshold
                    current_block.append(event)
                else:
                    if len(current_block) >= 3:  # Minimum block size
                        block_start = current_block[0].timestamp
                        block_end = current_block[-1].timestamp
                        block_duration = block_end - block_start
                        
                        if block_duration >= timedelta(hours=1):  # Minimum duration
                            avg_performance = np.mean([self._calculate_event_performance(e) 
                                                     for e in current_block])
                            
                            work_blocks.append({
                                'date': date.isoformat(),
                                'start_time': block_start.time().isoformat(),
                                'end_time': block_end.time().isoformat(),
                                'duration_hours': block_duration.total_seconds() / 3600,
                                'avg_performance': avg_performance,
                                'event_count': len(current_block)
                            })
                    
                    current_block = []
            
            # Process final block
            if len(current_block) >= 3:
                block_start = current_block[0].timestamp
                block_end = current_block[-1].timestamp
                block_duration = block_end - block_start
                
                if block_duration >= timedelta(hours=1):
                    avg_performance = np.mean([self._calculate_event_performance(e) 
                                             for e in current_block])
                    
                    work_blocks.append({
                        'date': date.isoformat(),
                        'start_time': block_start.time().isoformat(),
                        'end_time': block_end.time().isoformat(),
                        'duration_hours': block_duration.total_seconds() / 3600,
                        'avg_performance': avg_performance,
                        'event_count': len(current_block)
                    })
        
        # Sort by performance and return top blocks
        work_blocks.sort(key=lambda b: b['avg_performance'], reverse=True)
        return work_blocks[:20]  # Top 20 work blocks
    
    def _load_data(self):
        """Load temporal data from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # Load events
                for event_data in data.get('events', []):
                    event = TemporalEvent(
                        timestamp=datetime.fromisoformat(event_data['timestamp']),
                        event_type=event_data['event_type'],
                        duration=timedelta(seconds=event_data.get('duration_seconds', 0)) if event_data.get('duration_seconds') else None,
                        energy_level=event_data.get('energy_level'),
                        focus_level=event_data.get('focus_level'),
                        task_complexity=TaskComplexity(event_data['task_complexity']) if event_data.get('task_complexity') else None,
                        completion_status=event_data.get('completion_status'),
                        interruptions=event_data.get('interruptions', 0),
                        metadata=event_data.get('metadata', {})
                    )
                    self.events.append(event)
                
                logger.info(f"Loaded {len(self.events)} temporal events")
                
        except Exception as e:
            logger.error(f"Error loading temporal data: {e}")
    
    def save_data(self):
        """Save temporal data to storage."""
        try:
            data = {
                'events': [],
                'patterns': []
            }
            
            # Save events
            for event in self.events:
                event_data = {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'duration_seconds': event.duration.total_seconds() if event.duration else None,
                    'energy_level': event.energy_level,
                    'focus_level': event.focus_level,
                    'task_complexity': event.task_complexity.value if event.task_complexity else None,
                    'completion_status': event.completion_status,
                    'interruptions': event.interruptions,
                    'metadata': event.metadata
                }
                data['events'].append(event_data)
            
            # Save patterns
            for pattern in self.patterns:
                pattern_data = {
                    'pattern_type': pattern.pattern_type.value,
                    'description': pattern.description,
                    'confidence': pattern.confidence,
                    'time_range': [pattern.time_range[0].isoformat(), pattern.time_range[1].isoformat()],
                    'days_of_week': pattern.days_of_week,
                    'peak_value': pattern.peak_value,
                    'trend_direction': pattern.trend_direction,
                    'metadata': pattern.metadata
                }
                data['patterns'].append(pattern_data)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Temporal data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving temporal data: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get temporal analysis statistics."""
        return {
            'total_events': len(self.events),
            'detected_patterns': len(self.patterns),
            'date_range': {
                'start': min(e.timestamp for e in self.events).isoformat() if self.events else None,
                'end': max(e.timestamp for e in self.events).isoformat() if self.events else None
            },
            'event_types': list(set(e.event_type for e in self.events)),
            'has_energy_data': any(e.energy_level is not None for e in self.events),
            'has_focus_data': any(e.focus_level is not None for e in self.events),
            'storage_path': str(self.storage_path)
        }
