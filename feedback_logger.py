import json
import os
import pandas as pd

FEEDBACK_FILE = "global_feedback.json"

# Load the feedback data from the JSON file
def load_global_feedback():
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as file:
                return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    return {}

# Save feedback data to the JSON file
def save_global_feedback(feedback_data):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(feedback_data, file)

# Log feedback and replace previous feedback for each movie
def log_feedback(movie_title, feedback_scores):
    feedback_data = load_global_feedback()
    feedback_data.update(feedback_scores)  # Update with latest feedback
    save_global_feedback(feedback_data)

# Retrieve feedback-adjusted recommendations
def filter_low_feedback(df, movie_title, threshold=5):
    feedback_data = load_global_feedback()

    # Apply feedback scores to the DataFrame
    df['feedback_score'] = df['title'].apply(lambda x: feedback_data.get(x, threshold))  # Default to threshold if no score
    high_feedback_movies = df[df['feedback_score'] >= threshold]

    return high_feedback_movies[df['title'] != movie_title]
