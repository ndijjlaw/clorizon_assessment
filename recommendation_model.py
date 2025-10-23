import pandas as pd
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json


class DataPreprocessor:
    # Handles data cleaning and user history extraction
    
    def preprocess(self, data: pd.DataFrame) -> Dict[str, List[Tuple]]:

        # Converts user behavior data into structured format.
        # Returns: { user_id: [(timestamp, item_id, action), ...] }
        
        user_history = defaultdict(list)
        
        for _, row in data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            action = row['action']
            timestamp = row['timestamp']
            
            user_history[user_id].append((timestamp, item_id, action))
        
        # Sort by timestamp to preserve sequential order
        for user in user_history:
            user_history[user].sort(key=lambda x: x[0])
        
        return user_history


class CooccurrenceModel:
   
    # Simple co-occurrence based recommendation model.
    # Tracks item transitions and supports incremental updates.
   
    
    def __init__(self, decay_factor: float = 1.0):
        
        self.co_matrix = defaultdict(lambda: defaultdict(float))
        self.item_counts = defaultdict(int)  # Track item popularity
        self.decay_factor = decay_factor
        self.version = "1.0.0"
        self.last_updated = None
    
    def train(self, user_history: Dict[str, List[Tuple]]) -> None:
        """Build co-occurrence matrix from user history"""
        for user, history in user_history.items():
            items = [record[1] for record in history]
            
            # Build co-occurrence pairs
            for i in range(len(items) - 1):
                item_A = items[i]
                item_B = items[i + 1]
                self.co_matrix[item_A][item_B] += 1.0
                self.item_counts[item_B] += 1
        
        self.last_updated = datetime.now()
    
    def incremental_update(self, new_interactions: List[Tuple[str, str, str, int]]) -> None:
        
        # Update model with new data without full retraining.
        # Critical for continuous learning systems.
        
        # Group by user and sort
        user_updates = defaultdict(list)
        for user_id, item_id, action, timestamp in new_interactions:
            user_updates[user_id].append((timestamp, item_id, action))
        
        # Sort each user's new interactions
        for user in user_updates:
            user_updates[user].sort(key=lambda x: x[0])
        
        # Update co-occurrence counts
        for user, history in user_updates.items():
            items = [record[1] for record in history]
            
            for i in range(len(items) - 1):
                item_A = items[i]
                item_B = items[i + 1]
                
                # Incremental update with optional decay
                self.co_matrix[item_A][item_B] += self.decay_factor
                self.item_counts[item_B] += 1
        
        self.last_updated = datetime.now()
        print(f"Model updated at {self.last_updated}")
    
    def recommend(self, last_item: str, k: int = 3, 
                  min_support: int = 1) -> List[Tuple[str, float]]:
        
        # Generate recommendations with confidence scores.
        if last_item not in self.co_matrix:
            # Cold start: return popular items
            return self._get_popular_items(k)
        
        related_items = self.co_matrix[last_item]
        
        # Filter by minimum support and sort by score
        candidates = [
            (item, score) 
            for item, score in related_items.items() 
            if score >= min_support
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:k]
    
    def _get_popular_items(self, k: int) -> List[Tuple[str, float]]:
        # Fallback: return most popular items
        popular = sorted(
            self.item_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        return popular
    
    def get_stats(self) -> Dict:
        """Return model statistics for monitoring"""
        return {
            "version": self.version,
            "last_updated": str(self.last_updated),
            "num_source_items": len(self.co_matrix),
            "num_target_items": len(self.item_counts),
            "total_transitions": sum(
                sum(targets.values()) 
                for targets in self.co_matrix.values()
            )
        }


# Example usage demonstrating incremental learning
if __name__ == "__main__":
    # Initial training data
    initial_data = pd.DataFrame([
        {"user_id": "U1", "item_id": "I1", "action": "viewed", "timestamp": 1},
        {"user_id": "U1", "item_id": "I2", "action": "clicked", "timestamp": 2},
        {"user_id": "U1", "item_id": "I3", "action": "purchased", "timestamp": 3},
        {"user_id": "U2", "item_id": "I2", "action": "viewed", "timestamp": 4},
        {"user_id": "U2", "item_id": "I3", "action": "clicked", "timestamp": 5},
    ])
    
    # Train initial model
    preprocessor = DataPreprocessor()
    user_history = preprocessor.preprocess(initial_data)
    
    model = CooccurrenceModel()
    model.train(user_history)
    
    print("Initial Model Stats:", model.get_stats())
    print(f"Recommendations after I2: {model.recommend('I2', k=3)}")
    
    # Simulate new interactions arriving
    new_interactions = [
        ("U3", "I1", "viewed", 6),
        ("U3", "I3", "clicked", 7),
        ("U1", "I2", "viewed", 8),
        ("U1", "I4", "purchased", 9),
    ]
    
    # Incremental update (no full retrain needed)
    model.incremental_update(new_interactions)
    
    print("\nUpdated Model Stats:", model.get_stats())
    print(f"Updated recommendations after I2: {model.recommend('I2', k=3)}")