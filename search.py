import streamlit as st
import requests

# TMDb API Key
TMDB_API_KEY = '879e5ded68168715edd7f39beb34441c'  # Replace with your actual TMDb API key
DEFAULT_POSTER_URL = "https://via.placeholder.com/200x300?text=No+Image"


# Function to fetch movie details from TMDb
def fetch_movie_details(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(url)
    return response.json()


def search_movie(df):
    st.header('Search for a Movie')
    movie_search = st.text_input('Enter a movie title')

    if movie_search:
        search_results = fetch_movie_details(movie_search)

        if 'results' in search_results and search_results['results']:
            for movie in search_results['results']:
                st.markdown(f"### {movie['title']}")
                poster_url = f"https://image.tmdb.org/t/p/w500{movie.get('poster_path', '')}" if movie.get(
                    'poster_path') else DEFAULT_POSTER_URL
                st.image(poster_url, width=200)
                st.markdown(f"**Rating:** {movie['vote_average']}/10")
                st.markdown(f"**Overview:** {movie['overview']}\n")
        else:
            st.write('No results found.')
