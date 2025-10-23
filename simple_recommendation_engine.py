import pandas as pd
from collections import defaultdict

# Step 1: Preprocess the Data

def preprocess(data: pd.DataFrame):
 
    # Converts user behavior data into a structured dictionary:{ user_id: [(timestamp, item_id, action), ...] }
   
    user_history = defaultdict(list)
    
    for _, row in data.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        action = row['action']
        timestamp = row['timestamp']
        
        user_history[user_id].append((timestamp, item_id, action))
    
    # Sort user actions by timestamp
    for user in user_history:
        user_history[user].sort(key=lambda x: x[0])
    
    return user_history


# Step 2: Build Co-occurrence Matrix

def build_cooccurrence_matrix(user_history):

    # Builds a mapping of item_A -> {item_B: count}, showing what items are likely to follow one another in user interactions.
    
    co_matrix = defaultdict(lambda: defaultdict(int))
    
    for user, history in user_history.items():
        items = [record[1] for record in history]
        
        for i in range(len(items) - 1):
            item_A = items[i]
            item_B = items[i + 1]
            co_matrix[item_A][item_B] += 1
            
    return co_matrix


# Step 3: Recommend Next Item

def recommend_next_item(last_item, co_matrix, k=3):

    # Recommend top-k likely next items based on co-occurrence counts.
    
    if last_item not in co_matrix:
        return []
    
    related_items = co_matrix[last_item]
    sorted_candidates = sorted(related_items.items(), key=lambda x: x[1], reverse=True)
    
    return [item for item, _ in sorted_candidates[:k]]


# Step 4: Example Pipeline

if __name__ == "__main__":
    # Sample user behavior data
    data = pd.DataFrame([
        {"user_id": "U1", "item_id": "I1", "action": "viewed", "timestamp": 1},
        {"user_id": "U1", "item_id": "I2", "action": "clicked", "timestamp": 2},
        {"user_id": "U1", "item_id": "I3", "action": "purchased", "timestamp": 3},
        {"user_id": "U2", "item_id": "I2", "action": "viewed", "timestamp": 4},
        {"user_id": "U2", "item_id": "I3", "action": "clicked", "timestamp": 5},
    ])
    
    user_history = preprocess(data)
    co_matrix = build_cooccurrence_matrix(user_history)
    
    last_item = "I2"
    recommendations = recommend_next_item(last_item, co_matrix)
    
    print(f"Next-step recommendations after '{last_item}': {recommendations}")
