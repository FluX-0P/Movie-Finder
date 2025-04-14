import streamlit as st
import pandas as pd

# Load your dataset
df = pd.read_csv('filtered_tmdb_movies.csv')  # Adjust the path as necessary


def recommend_by_genre(df):
    # Extract unique genres, removing any leading/trailing spaces
    unique_genres = set()

    for genre_list in df['genres'].dropna():
        # Split the genres and strip spaces
        genres = [genre.strip() for genre in genre_list.split(',')]
        unique_genres.update(genres)

    # Convert the set back to a sorted list
    genres_sorted = sorted(unique_genres)

    # Selectbox for genre selection
    selected_genre = st.selectbox('Select a genre:', genres_sorted)

    # Filter movies based on the selected genre
    filtered_movies = df[df['genres'].str.contains(selected_genre, case=False, na=False)]

    # Display the recommended movies
    st.write(f"Movies in the genre '{selected_genre}':")
    for index, row in filtered_movies.iterrows():
        st.write(f"**Title:** {row['title']}")
        st.write(f"**Overview:** {row['overview']}")
        st.image(row['poster_path'], width=200)  # Display the movie poster


# Main function to run the Streamlit app
def main():
    st.title("Movie Recommendation App")
    st.header("Recommend Movies by Genre")
    recommend_by_genre(df)


if __name__ == "__main__":
    main()
