import streamlit as st
import pandas as pd

# Import functionalities from different modules
from search import search_movie
from genre_recommend import recommend_by_genre
from mood_recommend import recommend_by_mood
from content_recommend import recommend_by_content
from feedback_logger import log_feedback

# Load your dataset here
@st.cache_data
def load_data():
    df = pd.read_csv('filtered_tmdb_movies.csv')  # Path to your TMDb dataset
    df['poster_path'] = df['poster_path'].apply(lambda x: f"https://image.tmdb.org/t/p/w500{x}" if pd.notna(x) else None)
    return df

# Main app function
def main():
    st.title("Movie Recommendation App")

    # Load the dataset
    df = load_data()

    # Dropdown for feature selection
    feature = st.selectbox("Choose a feature:", ["Search Movie", "Recommend by Genre", "Recommend by Mood", "Content-Based Recommendations"])

    if feature == "Search Movie":
        search_movie(df)
    elif feature == "Recommend by Genre":
        recommend_by_genre(df)
    elif feature == "Recommend by Mood":
        recommend_by_mood(df)
    elif feature == "Content-Based Recommendations":
        recommend_by_content(df)

if __name__ == "__main__":
    main()
