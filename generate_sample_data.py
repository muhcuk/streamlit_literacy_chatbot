import json
import os
from datetime import datetime, timedelta
import random

# We're in streamlit folder, so data is in ./data
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

print(f"🎲 Generating sample data...")

test_results = {"results": []}
feedback_data = {"feedback": []}

educations = ["Secondary School", "Diploma", "Bachelor's", "Master's"]
occupations = ["Student", "Employee", "Self-employed"]
genders = ["Male", "Female"]

for i in range(5):
    user_id = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d_%H%M%S")
    
    participant_info = {
        "age": random.randint(18, 30),
        "education": random.choice(educations),
        "gender": random.choice(genders),
        "occupation": random.choice(occupations)
    }
    
    # Pre-test
    pre_scores = {
        "Financial Knowledge": random.uniform(40, 70),
        "Financial Behavior": random.uniform(40, 70),
        "Financial Confidence": random.uniform(30, 60),
        "Financial Attitudes": random.uniform(50, 75),
    }
    pre_scores["Overall"] = sum(pre_scores.values()) / 4
    
    test_results["results"].append({
        "user_id": user_id,
        "timestamp": (datetime.now() - timedelta(days=i, hours=2)).isoformat(),
        "test_type": "pre",
        "participant_info": participant_info,
        "responses": [
            {"question_id": "Q1", "question": "Interest payment", "category": "Financial Knowledge", "response": "Option 2", "score": 2},
            {"question_id": "Q2", "question": "Budget", "category": "Financial Knowledge", "response": "Option 1", "score": 1}
        ],
        "scores": pre_scores
    })
    
    # Post-test (with improvement)
    improvement = random.uniform(10, 25)
    post_scores = {key: min(100, value + improvement) for key, value in pre_scores.items()}
    
    test_results["results"].append({
        "user_id": user_id,
        "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
        "test_type": "post",
        "participant_info": participant_info,
        "responses": [
            {"question_id": "Q1", "question": "Interest payment", "category": "Financial Knowledge", "response": "Option 3", "score": 3},
            {"question_id": "Q2", "question": "Budget", "category": "Financial Knowledge", "response": "Option 3", "score": 3}
        ],
        "scores": post_scores
    })
    
    # Feedback
    feedback_data["feedback"].append({
        "user_id": user_id,
        "timestamp": (datetime.now() - timedelta(days=i, hours=1)).isoformat(),
        "question": "How can I save money?",
        "answer": "Sample answer about saving money...",
        "rating": random.choice(["helpful", "helpful", "not_helpful"]),
        "sources_count": random.randint(3, 5)
    })

# Save files
with open(os.path.join(data_dir, "test_results.json"), 'w', encoding='utf-8') as f:
    json.dump(test_results, f, indent=2, ensure_ascii=False)

with open(os.path.join(data_dir, "user_feedback.json"), 'w', encoding='utf-8') as f:
    json.dump(feedback_data, f, indent=2, ensure_ascii=False)

print(f"✅ Generated {len(test_results['results'])} test results")
print(f"✅ Generated {len(feedback_data['feedback'])} feedback entries")
print(f"✅ Saved to: {os.path.abspath(data_dir)}/")