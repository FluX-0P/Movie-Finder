import streamlit as st
import pandas as pd

# Load your dataset
df = pd.read_csv('filtered_tmdb_movies.csv')  # Adjust the path as necessary


def recommend_by_mood(df):
    mood = st.selectbox('Select a mood:', ['Happy', 'Sad', 'Angry', 'Excited', 'Romantic'])  # Add more moods as needed

    # Define mood-based filtering logic
    mood_keywords = {
        'Happy': ['comedy', 'feel-good', 'funny', 'happy', 'light-hearted'],
        'Sad': ['drama', 'sad', 'emotional', 'heartbreaking', 'tragic'],
        'Angry': ['action', 'thriller', 'intense', 'violent'],
        'Excited': ['adventure', 'action', 'thriller'],
        'Romantic': ['romance', 'love', 'relationship', 'romantic']
    }

    # Get keywords for the selected mood
    keywords = mood_keywords.get(mood, [])

    # Filter movies based on mood keywords in the title or genres
    filtered_movies = df[
        df['title'].str.contains('|'.join(keywords), case=False, na=False) |
        df['genres'].str.contains('|'.join(keywords), case=False, na=False)
        ]

    # Display the recommended movies
    st.write(f"Movies for the mood '{mood}':")
    for index, row in filtered_movies.iterrows():
        st.write(f"**Title:** {row['title']}")
        st.write(f"**Overview:** {row['overview']}")  # Display the movie overview
        st.image(row['poster_path'], width=200)  # Display the movie poster


# Main function to run the Streamlit app
def main():
    st.title("Movie Recommendation App")

    # Mood-based recommendation section
    st.header("Recommend Movies by Mood")
    recommend_by_mood(df)


if __name__ == "__main__":
    main()
