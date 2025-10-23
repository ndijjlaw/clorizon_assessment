import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
import json


class DataIngestion:
    """Handles streaming data ingestion"""
    
    def __init__(self):
        self.buffer = deque(maxlen=10000)  # Ring buffer for recent events
    
    def ingest_event(self, user_id: str, item_id: str, 
                     action: str, timestamp: int) -> None:
        """Add single interaction event"""
        self.buffer.append({
            'user_id': user_id,
            'item_id': item_id,
            'action': action,
            'timestamp': timestamp
        })
    
    def get_batch(self, n: int = 100) -> List[Dict]:
        """Retrieve batch of recent events for processing"""
        batch_size = min(n, len(self.buffer))
        return list(self.buffer)[-batch_size:] if batch_size > 0 else []
    
    def load_from_stream(self, path: str) -> pd.DataFrame:
        """Load data from file (simulating stream)"""
        return pd.read_csv(path)


class FeatureStore:
    """Manages feature computation and caching"""
    
    def __init__(self):
        self.user_profiles = {}  # Cache user features
        self.item_features = {}  # Cache item features
    
    def compute_user_history(self, data: pd.DataFrame) -> Dict[str, list]:
        """Extract and structure user interaction sequences"""
        from collections import defaultdict
        user_history = defaultdict(list)
        
        for _, row in data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            action = row['action']
            timestamp = row['timestamp']
            
            user_history[user_id].append((timestamp, item_id, action))
        
        for user in user_history:
            user_history[user].sort(key=lambda x: x[0])
        
        return user_history
    
    def update_user_profile(self, user_id: str, recent_items: List[str]) -> None:
        """Cache user's recent interaction pattern"""
        self.user_profiles[user_id] = {
            'recent_items': recent_items[-10:],  # Last 10 items
            'updated_at': datetime.now()
        }


class ModelTraining:
    """Handles model training and updates"""
    
    def __init__(self, model):
        self.model = model
        self.training_history = []
    
    def train(self, user_history: Dict[str, list]) -> None:
        """Initial training from historical data"""
        self.model.train(user_history)
        self.training_history.append({
            'timestamp': datetime.now(),
            'type': 'full_train',
            'stats': self.model.get_stats()
        })
    
    def incremental_update(self, new_interactions: List[Tuple]) -> None:
        """Update model with new data"""
        self.model.incremental_update(new_interactions)
        self.training_history.append({
            'timestamp': datetime.now(),
            'type': 'incremental',
            'stats': self.model.get_stats()
        })


class RecommendationEngine:
    """Serves recommendations with fallback strategies"""
    
    def __init__(self, model):
        self.model = model
        self.cache = {}  # LRU cache for frequent queries
        self.cache_ttl = timedelta(minutes=5)
        self.fallback_items = ["I1", "I2", "I3"]  # Popular items as fallback
    
    def recommend(self, last_item: str, k: int = 3, 
                  use_cache: bool = True) -> List[Tuple[str, float]]:
        """
        Generate recommendations with caching and fallback.
        
        Returns:
            List of (item_id, score) tuples
        """
        # Check cache first
        cache_key = f"{last_item}_{k}"
        if use_cache and cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        try:
            # Get recommendations from model
            recommendations = self.model.recommend(last_item, k)
            
            # Apply business rules/filters here
            recommendations = self._apply_filters(recommendations)
            
            # Add diversity/exploration
            recommendations = self._add_exploration(recommendations, k)
            
            # Cache result
            self.cache[cache_key] = (recommendations, datetime.now())
            
            return recommendations
        
        except Exception as e:
            print(f"Recommendation error: {e}")
            return self._fallback_recommend(k)
    
    def _apply_filters(self, recommendations: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Apply business rules (e.g., remove out-of-stock items)"""
        # TODO: Implement filtering logic based on business rules
        return recommendations
    
    def _add_exploration(self, recommendations: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
        """
        Add diversity to avoid filter bubbles.
        Use epsilon-greedy: occasionally show random items.
        """
        import random
        if random.random() < 0.1 and len(recommendations) > 0:  # 10% exploration
            # Replace last item with random exploration
            recommendations = recommendations[:-1]
            # Add random item (in production, sample from item catalog)
        return recommendations
    
    def _fallback_recommend(self, k: int) -> List[Tuple[str, float]]:
        """Fallback when primary recommendation fails"""
        return [(item, 1.0) for item in self.fallback_items[:k]]


class FeedbackLoop:
    """
    Captures user feedback and triggers model updates.
    This is the KEY component for continuous learning.
    """
    
    def __init__(self, trainer: ModelTraining, monitor: 'ModelMonitor'):
        self.trainer = trainer
        self.monitor = monitor
        self.feedback_buffer = []
        self.update_threshold = 50  # Retrain after N new interactions
        self.update_interval = timedelta(hours=1)  # Or time-based
        self.last_update = datetime.now()
    
    def collect_feedback(self, user_id: str, last_item: str,
                        shown_items: List[str], 
                        clicked_items: List[str],
                        timestamp: int) -> None:
        """
        Collect user response to recommendations.
        
        Args:
            user_id: User identifier
            last_item: Item that triggered recommendations
            shown_items: Items recommended to user
            clicked_items: Items user actually clicked
            timestamp: When interaction occurred
        """
        feedback = {
            'user_id': user_id,
            'last_item': last_item,
            'shown_items': shown_items,
            'clicked_items': clicked_items,
            'timestamp': timestamp,
            'click_rate': len(clicked_items) / len(shown_items) if shown_items else 0
        }
        
        self.feedback_buffer.append(feedback)
        
        # Log for monitoring
        self.monitor.log_prediction(
            shown_items=shown_items,
            actual_clicks=clicked_items,
            timestamp=timestamp
        )
        
        # Check if we should trigger retraining
        if self._should_retrain():
            self._retrain()
    
    def _should_retrain(self) -> bool:
        """
        Decide when to update model. Multiple strategies:
        1. Volume-based: After N new interactions
        2. Time-based: Every X hours
        3. Performance-based: When metrics degrade
        """
        # Volume-based check
        if len(self.feedback_buffer) >= self.update_threshold:
            return True
        
        # Time-based check
        if datetime.now() - self.last_update >= self.update_interval:
            return True
        
        # Performance-based check
        recent_performance = self.monitor.get_recent_metrics()
        if recent_performance and recent_performance['click_rate'] < 0.05:
            print("Performance degradation detected, triggering retrain")
            return True
        
        return False
    
    def _retrain(self) -> None:
        """
        Update model with accumulated feedback.
        Convert feedback into training format and update incrementally.
        """
        print(f"Retraining with {len(self.feedback_buffer)} new interactions...")
        
        # Convert feedback to interaction format
        new_interactions = []
        for fb in self.feedback_buffer:
            # Only add clicked items as positive signals
            for clicked_item in fb['clicked_items']:
                new_interactions.append((
                    fb['user_id'],
                    clicked_item,
                    'clicked',
                    fb['timestamp']
                ))
        
        if new_interactions:
            # Incremental update (not full retrain)
            self.trainer.incremental_update(new_interactions)
            
            # Clear buffer after successful update
            self.feedback_buffer.clear()
            self.last_update = datetime.now()
            
            print(f"Model updated successfully at {self.last_update}")


class ModelMonitor:
    """Tracks model performance and detects drift"""
    
    def __init__(self):
        self.prediction_log = deque(maxlen=1000)  # Last 1000 predictions
        self.metrics_history = []
    
    def log_prediction(self, shown_items: List[str], 
                      actual_clicks: List[str],
                      timestamp: int) -> None:
        """Log each prediction for analysis"""
        self.prediction_log.append({
            'shown': shown_items,
            'clicked': actual_clicks,
            'timestamp': timestamp,
            'hit': any(item in actual_clicks for item in shown_items)
        })
    
    def get_recent_metrics(self, window: int = 100) -> Optional[Dict]:
        """Calculate metrics over recent predictions"""
        if len(self.prediction_log) < window:
            return None
        
        recent = list(self.prediction_log)[-window:]
        
        total_hits = sum(1 for pred in recent if pred['hit'])
        total_clicks = sum(len(pred['clicked']) for pred in recent)
        total_shown = sum(len(pred['shown']) for pred in recent)
        
        metrics = {
            'hit_rate': total_hits / window if window > 0 else 0,
            'click_rate': total_clicks / total_shown if total_shown > 0 else 0,
            'window_size': window,
            'timestamp': datetime.now()
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def detect_drift(self, threshold: float = 0.2) -> bool:
        """
        Detect if model performance has degraded significantly.
        Compare recent performance to historical baseline.
        """
        if len(self.metrics_history) < 2:
            return False
        
        current = self.metrics_history[-1]['hit_rate']
        baseline = sum(m['hit_rate'] for m in self.metrics_history[:-1]) / (len(self.metrics_history) - 1)
        
        if baseline - current > threshold:
            print(f"DRIFT DETECTED: Current {current:.2%} vs Baseline {baseline:.2%}")
            return True
        
        return False


# ---------------------------------------------------
# ORCHESTRATION: Putting it all together
# ----------------------------------------

if __name__ == "__main__":
    from recommendation_model import CooccurrenceModel, DataPreprocessor
    
    # Initialize all components
    data_ingestion = DataIngestion()
    feature_store = FeatureStore()
    model = CooccurrenceModel()
    trainer = ModelTraining(model)
    monitor = ModelMonitor()
    recommender = RecommendationEngine(model)
    feedback_loop = FeedbackLoop(trainer, monitor)
    
    # Initial training
    initial_data = pd.DataFrame([
        {"user_id": "U1", "item_id": "I1", "action": "viewed", "timestamp": 1},
        {"user_id": "U1", "item_id": "I2", "action": "clicked", "timestamp": 2},
        {"user_id": "U1", "item_id": "I3", "action": "purchased", "timestamp": 3},
        {"user_id": "U2", "item_id": "I2", "action": "viewed", "timestamp": 4},
        {"user_id": "U2", "item_id": "I3", "action": "clicked", "timestamp": 5},
    ])
    
    user_history = feature_store.compute_user_history(initial_data)
    trainer.train(user_history)
    
    print("=" * 50)
    print("INITIAL MODEL STATS")
    print("=" * 50)
    print(json.dumps(model.get_stats(), indent=2))
    
    # Simulate continuous operation
    print("\n" + "=" * 50)
    print("SERVING RECOMMENDATIONS")
    print("=" * 50)
    
    # User requests recommendation
    last_item = "I2"
    recommendations = recommender.recommend(last_item, k=3)
    print(f"User's last item: {last_item}")
    print(f"Recommended: {recommendations}")
    
    # User interacts (clicks on some recommendations)
    clicked = ["I3"]  # User clicked on I3
    shown = [rec[0] for rec in recommendations]
    
    # Collect feedback
    feedback_loop.collect_feedback(
        user_id="U1",
        last_item=last_item,
        shown_items=shown,
        clicked_items=clicked,
        timestamp=10
    )
    
    # Simulate more interactions
    print("\n" + "=" * 50)
    print("SIMULATING CONTINUOUS FEEDBACK")
    print("=" * 50)
    
    for i in range(60):  # Simulate 60 interactions
        user_id = f"U{(i % 3) + 1}"
        last = f"I{(i % 4) + 1}"
        
        recs = recommender.recommend(last, k=3)
        shown = [rec[0] for rec in recs]
        
        # Simulate user clicking with 20% probability
        import random
        clicked = [shown[0]] if random.random() < 0.2 and shown else []
        
        feedback_loop.collect_feedback(
            user_id=user_id,
            last_item=last,
            shown_items=shown,
            clicked_items=clicked,
            timestamp=100 + i
        )
    
    print("\n" + "=" * 50)
    print("FINAL MODEL STATS")
    print("=" * 50)
    print(json.dumps(model.get_stats(), indent=2))
    
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    recent_metrics = monitor.get_recent_metrics(window=50)
    if recent_metrics:
        print(json.dumps({k: v for k, v in recent_metrics.items() if k != 'timestamp'}, indent=2))