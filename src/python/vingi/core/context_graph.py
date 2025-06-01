"""
Context Graph Management for Personal Knowledge Representation

This module implements a sophisticated context graph that maintains relationships
between user activities, preferences, patterns, and external information.
"""

import json
import sqlite3
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the context graph."""
    PERSON = "person"
    LOCATION = "location"
    ACTIVITY = "activity"
    PREFERENCE = "preference"
    PATTERN = "pattern"
    TASK = "task"
    DOMAIN = "domain"
    RESOURCE = "resource"
    INSIGHT = "insight"
    AUTOMATION_RULE = "automation_rule"


class RelationshipType(Enum):
    """Types of relationships between nodes."""
    LIKES = "likes"
    DISLIKES = "dislikes"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    INFLUENCES = "influences"
    OCCURRED_AT = "occurred_at"
    PERFORMED_BY = "performed_by"
    DEPENDS_ON = "depends_on"
    CONFLICTS_WITH = "conflicts_with"
    SIMILAR_TO = "similar_to"
    TRIGGERS = "triggers"
    CONTAINS = "contains"


@dataclass
class ContextNode:
    """Represents a node in the context graph."""
    id: str
    node_type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'id': self.id,
            'node_type': self.node_type.value,
            'name': self.name,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'relevance_score': self.relevance_score,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextNode':
        """Create node from dictionary."""
        return cls(
            id=data['id'],
            node_type=NodeType(data['node_type']),
            name=data['name'],
            properties=data.get('properties', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            relevance_score=data.get('relevance_score', 1.0),
            confidence=data.get('confidence', 1.0)
        )


@dataclass
class ContextRelationship:
    """Represents a relationship between nodes."""
    id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship_type': self.relationship_type.value,
            'strength': self.strength,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'last_reinforced': self.last_reinforced.isoformat(),
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextRelationship':
        """Create relationship from dictionary."""
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            relationship_type=RelationshipType(data['relationship_type']),
            strength=data.get('strength', 1.0),
            properties=data.get('properties', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            last_reinforced=datetime.fromisoformat(data['last_reinforced']),
            confidence=data.get('confidence', 1.0)
        )


@dataclass
class ContextQuery:
    """Query for context graph retrieval."""
    node_types: Optional[List[NodeType]] = None
    relationship_types: Optional[List[RelationshipType]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    relevance_threshold: float = 0.1
    max_results: int = 100
    include_properties: bool = True
    keywords: Optional[List[str]] = None


class ContextGraphManager:
    """
    Manages the user's personal context graph with privacy-first storage.
    
    This class maintains a sophisticated graph of user preferences, activities,
    patterns, and relationships to enable intelligent automation and assistance.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the context graph manager.
        
        Args:
            storage_path: Path to SQLite database file for persistent storage
        """
        self.storage_path = storage_path or Path.home() / "Library" / "Application Support" / "Vingi" / "context.db"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory graph for fast access
        self.nodes: Dict[str, ContextNode] = {}
        self.relationships: Dict[str, ContextRelationship] = {}
        self.node_relationships: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.last_optimization = datetime.now()
        self.query_cache: Dict[str, Tuple[datetime, Any]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Initialize database
        self._init_database()
        self._load_from_storage()
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            
            # Nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    properties TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0
                )
            """)
            
            # Relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    properties TEXT,
                    created_at TEXT NOT NULL,
                    last_reinforced TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (source_id) REFERENCES nodes (id),
                    FOREIGN KEY (target_id) REFERENCES nodes (id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (node_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_updated ON nodes (updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships (source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships (target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships (relationship_type)")
            
            conn.commit()
    
    def _load_from_storage(self):
        """Load graph data from persistent storage."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                
                # Load nodes
                cursor.execute("SELECT * FROM nodes")
                for row in cursor.fetchall():
                    node_data = {
                        'id': row[0],
                        'node_type': row[1],
                        'name': row[2],
                        'properties': json.loads(row[3]) if row[3] else {},
                        'created_at': row[4],
                        'updated_at': row[5],
                        'relevance_score': row[6],
                        'confidence': row[7]
                    }
                    node = ContextNode.from_dict(node_data)
                    self.nodes[node.id] = node
                
                # Load relationships
                cursor.execute("SELECT * FROM relationships")
                for row in cursor.fetchall():
                    rel_data = {
                        'id': row[0],
                        'source_id': row[1],
                        'target_id': row[2],
                        'relationship_type': row[3],
                        'strength': row[4],
                        'properties': json.loads(row[5]) if row[5] else {},
                        'created_at': row[6],
                        'last_reinforced': row[7],
                        'confidence': row[8]
                    }
                    rel = ContextRelationship.from_dict(rel_data)
                    self.relationships[rel.id] = rel
                    self.node_relationships[rel.source_id].add(rel.id)
                    self.node_relationships[rel.target_id].add(rel.id)
                
                logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.relationships)} relationships")
                
        except Exception as e:
            logger.error(f"Error loading from storage: {e}")
    
    def _save_to_storage(self):
        """Save graph data to persistent storage."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.cursor()
                
                # Save nodes
                for node in self.nodes.values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO nodes 
                        (id, node_type, name, properties, created_at, updated_at, relevance_score, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        node.id,
                        node.node_type.value,
                        node.name,
                        json.dumps(node.properties),
                        node.created_at.isoformat(),
                        node.updated_at.isoformat(),
                        node.relevance_score,
                        node.confidence
                    ))
                
                # Save relationships
                for rel in self.relationships.values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO relationships 
                        (id, source_id, target_id, relationship_type, strength, properties, created_at, last_reinforced, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rel.id,
                        rel.source_id,
                        rel.target_id,
                        rel.relationship_type.value,
                        rel.strength,
                        json.dumps(rel.properties),
                        rel.created_at.isoformat(),
                        rel.last_reinforced.isoformat(),
                        rel.confidence
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving to storage: {e}")
    
    def add_node(self, node: ContextNode) -> str:
        """
        Add a node to the context graph.
        
        Args:
            node: The node to add
            
        Returns:
            The node ID
        """
        if not node.id:
            node.id = self._generate_node_id(node)
        
        node.updated_at = datetime.now()
        self.nodes[node.id] = node
        
        logger.debug(f"Added node: {node.name} ({node.node_type.value})")
        return node.id
    
    def add_relationship(self, relationship: ContextRelationship) -> str:
        """
        Add a relationship to the context graph.
        
        Args:
            relationship: The relationship to add
            
        Returns:
            The relationship ID
        """
        if not relationship.id:
            relationship.id = self._generate_relationship_id(relationship)
        
        # Verify source and target nodes exist
        if relationship.source_id not in self.nodes:
            raise ValueError(f"Source node {relationship.source_id} not found")
        if relationship.target_id not in self.nodes:
            raise ValueError(f"Target node {relationship.target_id} not found")
        
        relationship.last_reinforced = datetime.now()
        self.relationships[relationship.id] = relationship
        self.node_relationships[relationship.source_id].add(relationship.id)
        self.node_relationships[relationship.target_id].add(relationship.id)
        
        logger.debug(f"Added relationship: {relationship.relationship_type.value} "
                    f"({relationship.source_id} -> {relationship.target_id})")
        return relationship.id
    
    def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a node in the context graph.
        
        Args:
            node_id: ID of the node to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if update was successful
        """
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        # Update allowed fields
        if 'name' in updates:
            node.name = updates['name']
        if 'properties' in updates:
            node.properties.update(updates['properties'])
        if 'relevance_score' in updates:
            node.relevance_score = updates['relevance_score']
        if 'confidence' in updates:
            node.confidence = updates['confidence']
        
        node.updated_at = datetime.now()
        
        logger.debug(f"Updated node: {node_id}")
        return True
    
    def get_node(self, node_id: str) -> Optional[ContextNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[ContextNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_related_nodes(self, node_id: str, 
                         relationship_types: Optional[List[RelationshipType]] = None,
                         max_depth: int = 1) -> List[Tuple[ContextNode, ContextRelationship, int]]:
        """
        Get nodes related to a given node.
        
        Args:
            node_id: ID of the source node
            relationship_types: Optional filter for relationship types
            max_depth: Maximum depth to traverse (1 = direct relationships only)
            
        Returns:
            List of (node, relationship, depth) tuples
        """
        if node_id not in self.nodes:
            return []
        
        visited = set()
        queue = deque([(node_id, None, 0)])  # (node_id, relationship, depth)
        results = []
        
        while queue:
            current_id, rel, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Add current node to results (except the starting node)
            if depth > 0:
                results.append((self.nodes[current_id], rel, depth))
            
            # Add related nodes to queue
            if depth < max_depth:
                for rel_id in self.node_relationships.get(current_id, set()):
                    relationship = self.relationships[rel_id]
                    
                    # Filter by relationship type if specified
                    if relationship_types and relationship.relationship_type not in relationship_types:
                        continue
                    
                    # Add the other node in the relationship
                    other_id = relationship.target_id if relationship.source_id == current_id else relationship.source_id
                    
                    if other_id not in visited:
                        queue.append((other_id, relationship, depth + 1))
        
        return results
    
    def find_nodes(self, query: ContextQuery) -> List[ContextNode]:
        """
        Find nodes matching a query.
        
        Args:
            query: The search query
            
        Returns:
            List of matching nodes
        """
        # Check cache first
        cache_key = self._generate_query_cache_key(query)
        if cache_key in self.query_cache:
            cached_time, cached_result = self.query_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        results = []
        
        for node in self.nodes.values():
            # Filter by node types
            if query.node_types and node.node_type not in query.node_types:
                continue
            
            # Filter by relevance threshold
            if node.relevance_score < query.relevance_threshold:
                continue
            
            # Filter by time range
            if query.time_range:
                start_time, end_time = query.time_range
                if not (start_time <= node.updated_at <= end_time):
                    continue
            
            # Filter by keywords
            if query.keywords:
                text_content = f"{node.name} {json.dumps(node.properties)}"
                if not any(keyword.lower() in text_content.lower() for keyword in query.keywords):
                    continue
            
            results.append(node)
        
        # Sort by relevance score
        results.sort(key=lambda n: n.relevance_score, reverse=True)
        
        # Apply max results limit
        results = results[:query.max_results]
        
        # Cache the result
        self.query_cache[cache_key] = (datetime.now(), results)
        
        return results
    
    def add_activity_context(self, activity_name: str, domain: str, 
                           properties: Dict[str, Any]) -> str:
        """
        Add an activity with full context.
        
        Args:
            activity_name: Name of the activity
            domain: Domain the activity belongs to
            properties: Additional properties
            
        Returns:
            Activity node ID
        """
        # Create activity node
        activity_node = ContextNode(
            id="",
            node_type=NodeType.ACTIVITY,
            name=activity_name,
            properties=properties
        )
        activity_id = self.add_node(activity_node)
        
        # Create or get domain node
        domain_nodes = [n for n in self.get_nodes_by_type(NodeType.DOMAIN) if n.name == domain]
        if domain_nodes:
            domain_id = domain_nodes[0].id
        else:
            domain_node = ContextNode(
                id="",
                node_type=NodeType.DOMAIN,
                name=domain,
                properties={}
            )
            domain_id = self.add_node(domain_node)
        
        # Create relationship
        relationship = ContextRelationship(
            id="",
            source_id=activity_id,
            target_id=domain_id,
            relationship_type=RelationshipType.PART_OF,
            strength=1.0
        )
        self.add_relationship(relationship)
        
        return activity_id
    
    def record_preference(self, item: str, preference_type: str, 
                         strength: float, context: Dict[str, Any]) -> str:
        """
        Record a user preference.
        
        Args:
            item: The item being rated
            preference_type: Type of preference (like/dislike)
            strength: Strength of preference (0.0 to 1.0)
            context: Additional context
            
        Returns:
            Preference node ID
        """
        preference_node = ContextNode(
            id="",
            node_type=NodeType.PREFERENCE,
            name=f"{preference_type}:{item}",
            properties={
                'item': item,
                'preference_type': preference_type,
                'strength': strength,
                'context': context
            }
        )
        
        return self.add_node(preference_node)
    
    def get_user_preferences(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user preferences, optionally filtered by domain.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            Dictionary of preferences
        """
        preferences = self.get_nodes_by_type(NodeType.PREFERENCE)
        
        if domain:
            # Filter preferences related to the domain
            domain_nodes = [n for n in self.get_nodes_by_type(NodeType.DOMAIN) if n.name == domain]
            if domain_nodes:
                domain_id = domain_nodes[0].id
                related_prefs = []
                for pref in preferences:
                    related = self.get_related_nodes(pref.id, max_depth=2)
                    if any(node.id == domain_id for node, _, _ in related):
                        related_prefs.append(pref)
                preferences = related_prefs
        
        # Organize preferences
        result = {
            'likes': [],
            'dislikes': [],
            'neutral': []
        }
        
        for pref in preferences:
            pref_type = pref.properties.get('preference_type', 'neutral')
            strength = pref.properties.get('strength', 0.5)
            item = pref.properties.get('item', pref.name)
            
            entry = {
                'item': item,
                'strength': strength,
                'context': pref.properties.get('context', {}),
                'updated': pref.updated_at.isoformat()
            }
            
            if pref_type == 'like':
                result['likes'].append(entry)
            elif pref_type == 'dislike':
                result['dislikes'].append(entry)
            else:
                result['neutral'].append(entry)
        
        # Sort by strength
        for category in result.values():
            category.sort(key=lambda x: x['strength'], reverse=True)
        
        return result
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in the context graph.
        
        Returns:
            Dictionary containing pattern analysis results
        """
        analysis = {
            'node_distribution': self._analyze_node_distribution(),
            'relationship_patterns': self._analyze_relationship_patterns(),
            'temporal_patterns': self._analyze_temporal_patterns(),
            'preference_trends': self._analyze_preference_trends(),
            'activity_clusters': self._analyze_activity_clusters()
        }
        
        return analysis
    
    def _analyze_node_distribution(self) -> Dict[str, int]:
        """Analyze distribution of node types."""
        distribution = defaultdict(int)
        for node in self.nodes.values():
            distribution[node.node_type.value] += 1
        return dict(distribution)
    
    def _analyze_relationship_patterns(self) -> Dict[str, Any]:
        """Analyze relationship patterns."""
        rel_types = defaultdict(int)
        avg_strength = defaultdict(list)
        
        for rel in self.relationships.values():
            rel_types[rel.relationship_type.value] += 1
            avg_strength[rel.relationship_type.value].append(rel.strength)
        
        # Calculate average strengths
        avg_strength_final = {}
        for rel_type, strengths in avg_strength.items():
            avg_strength_final[rel_type] = np.mean(strengths) if strengths else 0.0
        
        return {
            'relationship_counts': dict(rel_types),
            'average_strengths': avg_strength_final,
            'total_relationships': len(self.relationships)
        }
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in the graph."""
        now = datetime.now()
        time_buckets = {
            'last_hour': [],
            'last_day': [],
            'last_week': [],
            'last_month': []
        }
        
        for node in self.nodes.values():
            time_diff = now - node.updated_at
            
            if time_diff <= timedelta(hours=1):
                time_buckets['last_hour'].append(node)
            if time_diff <= timedelta(days=1):
                time_buckets['last_day'].append(node)
            if time_diff <= timedelta(weeks=1):
                time_buckets['last_week'].append(node)
            if time_diff <= timedelta(days=30):
                time_buckets['last_month'].append(node)
        
        return {
            'activity_counts': {period: len(nodes) for period, nodes in time_buckets.items()},
            'most_active_types': self._get_most_active_types(time_buckets['last_week'])
        }
    
    def _analyze_preference_trends(self) -> Dict[str, Any]:
        """Analyze preference trends."""
        preferences = self.get_nodes_by_type(NodeType.PREFERENCE)
        
        likes = [p for p in preferences if p.properties.get('preference_type') == 'like']
        dislikes = [p for p in preferences if p.properties.get('preference_type') == 'dislike']
        
        return {
            'total_preferences': len(preferences),
            'likes_vs_dislikes': {'likes': len(likes), 'dislikes': len(dislikes)},
            'average_preference_strength': np.mean([p.properties.get('strength', 0.5) for p in preferences]) if preferences else 0.5,
            'recent_preferences': len([p for p in preferences if (datetime.now() - p.updated_at).days <= 7])
        }
    
    def _analyze_activity_clusters(self) -> Dict[str, Any]:
        """Analyze activity clustering by domain."""
        activities = self.get_nodes_by_type(NodeType.ACTIVITY)
        domain_clusters = defaultdict(list)
        
        for activity in activities:
            # Find related domains
            related = self.get_related_nodes(activity.id, [RelationshipType.PART_OF])
            domains = [node.name for node, _, _ in related if node.node_type == NodeType.DOMAIN]
            
            for domain in domains:
                domain_clusters[domain].append(activity)
        
        cluster_analysis = {}
        for domain, domain_activities in domain_clusters.items():
            cluster_analysis[domain] = {
                'activity_count': len(domain_activities),
                'recent_activity': len([a for a in domain_activities if (datetime.now() - a.updated_at).days <= 7]),
                'average_relevance': np.mean([a.relevance_score for a in domain_activities]) if domain_activities else 0.0
            }
        
        return cluster_analysis
    
    def _get_most_active_types(self, nodes: List[ContextNode]) -> Dict[str, int]:
        """Get most active node types in a set of nodes."""
        type_counts = defaultdict(int)
        for node in nodes:
            type_counts[node.node_type.value] += 1
        
        # Sort by count and return top 5
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_types[:5])
    
    def _generate_node_id(self, node: ContextNode) -> str:
        """Generate a unique ID for a node."""
        content = f"{node.node_type.value}:{node.name}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_relationship_id(self, relationship: ContextRelationship) -> str:
        """Generate a unique ID for a relationship."""
        content = f"{relationship.source_id}:{relationship.target_id}:{relationship.relationship_type.value}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_query_cache_key(self, query: ContextQuery) -> str:
        """Generate a cache key for a query."""
        query_str = json.dumps(asdict(query), default=str, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def optimize_graph(self):
        """Optimize the graph by removing stale data and updating relevance scores."""
        now = datetime.now()
        
        # Decay relevance scores over time
        for node in self.nodes.values():
            age_days = (now - node.updated_at).days
            decay_factor = max(0.1, 1.0 - (age_days * 0.01))  # 1% decay per day, minimum 0.1
            node.relevance_score *= decay_factor
        
        # Remove very old, low-relevance nodes
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            age_days = (now - node.updated_at).days
            if age_days > 365 and node.relevance_score < 0.1:  # Older than 1 year and very low relevance
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            self.remove_node(node_id)
        
        # Clear old cache entries
        cache_keys_to_remove = []
        for cache_key, (cached_time, _) in self.query_cache.items():
            if now - cached_time > self.cache_ttl:
                cache_keys_to_remove.append(cache_key)
        
        for cache_key in cache_keys_to_remove:
            del self.query_cache[cache_key]
        
        self.last_optimization = now
        logger.info(f"Graph optimization complete. Removed {len(nodes_to_remove)} nodes and {len(cache_keys_to_remove)} cache entries.")
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its relationships."""
        if node_id not in self.nodes:
            return False
        
        # Remove all relationships involving this node
        relationships_to_remove = []
        for rel_id in self.node_relationships.get(node_id, set()):
            relationships_to_remove.append(rel_id)
        
        for rel_id in relationships_to_remove:
            self.remove_relationship(rel_id)
        
        # Remove the node
        del self.nodes[node_id]
        if node_id in self.node_relationships:
            del self.node_relationships[node_id]
        
        logger.debug(f"Removed node: {node_id}")
        return True
    
    def remove_relationship(self, relationship_id: str) -> bool:
        """Remove a relationship."""
        if relationship_id not in self.relationships:
            return False
        
        relationship = self.relationships[relationship_id]
        
        # Remove from node relationships
        self.node_relationships[relationship.source_id].discard(relationship_id)
        self.node_relationships[relationship.target_id].discard(relationship_id)
        
        # Remove the relationship
        del self.relationships[relationship_id]
        
        logger.debug(f"Removed relationship: {relationship_id}")
        return True
    
    def save(self):
        """Save the context graph to persistent storage."""
        self._save_to_storage()
        logger.info("Context graph saved to storage")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'node_count': len(self.nodes),
            'relationship_count': len(self.relationships),
            'node_types': self._analyze_node_distribution(),
            'last_optimization': self.last_optimization.isoformat(),
            'cache_size': len(self.query_cache),
            'storage_path': str(self.storage_path)
        }
