from typing import List, Dict
from simple_recommendation_engine import *

class DataIngestion:
    def load_stream(self, path: str) -> pd.DataFrame:
        # Load user behavior data
        return pd.read_csv(path)


class FeatureStore:
    def compute_user_history(self, data: pd.DataFrame) -> Dict[str, list]:
        # Extract and sort user interaction history
        return preprocess(data)


class ModelTraining:
    def train(self, user_history: Dict[str, list]):
        # Train/update the co-occurrence model
        return build_cooccurrence_matrix(user_history)


class RecommendationEngine:
    def __init__(self, model):
        self.model = model
    
    def recommend(self, last_item: str, k: int = 3) -> List[str]:
        # Generate recommendations
        return recommend_next_item(last_item, self.model, k)


class FeedbackLoop:
    def collect_feedback(self, user_id: str, shown_items: List[str], clicked_items: List[str]):
        # Collect and log user feedback. In a real system, this data would be stored and used for retraining.
        feedback = {
            "user_id": user_id,
            "shown_items": shown_items,
            "clicked_items": clicked_items
        }
        print("Feedback received:", feedback)
        return feedback


# -------------------------------
# Orchestrating the modules
# -------------------------------

if __name__ == "__main__":
    # Initialize components
    data_ingestion = DataIngestion()
    feature_store = FeatureStore()
    trainer = ModelTraining()
    
    # Load and prepare data
    data = pd.DataFrame([
        {"user_id": "U1", "item_id": "I1", "action": "viewed", "timestamp": 1},
        {"user_id": "U1", "item_id": "I2", "action": "clicked", "timestamp": 2},
        {"user_id": "U1", "item_id": "I3", "action": "purchased", "timestamp": 3},
        {"user_id": "U2", "item_id": "I2", "action": "viewed", "timestamp": 4},
        {"user_id": "U2", "item_id": "I3", "action": "clicked", "timestamp": 5},
    ])
    
    user_history = feature_store.compute_user_history(data)
    co_matrix = trainer.train(user_history)
    
    recommender = RecommendationEngine(co_matrix)
    feedback_loop = FeedbackLoop()
    
    # Recommend and simulate feedback
    last_item = "I2"
    recs = recommender.recommend(last_item)
    print(f"Recommended next items: {recs}")
    
    feedback_loop.collect_feedback(user_id="U1", shown_items=recs, clicked_items=["I3"])
