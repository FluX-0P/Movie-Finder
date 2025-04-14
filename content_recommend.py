import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from feedback_logger import log_feedback, filter_low_feedback

def recommend_by_content(df):
    movie_title = st.text_input("Enter a movie title:").strip()

    if movie_title:
        # Normalize titles for easier comparison
        df['normalized_title'] = df['title'].str.strip().str.lower()

        if movie_title.lower() not in df['normalized_title'].values:
            st.error("Movie not found in the dataset.")
            return

        # Retrieve overview and genres
        movie_overview = df[df['normalized_title'] == movie_title.lower()]['overview'].values[0]
        movie_genres = df[df['normalized_title'] == movie_title.lower()]['genres'].values[0]
        movie_genres = [genre.strip() for genre in movie_genres.split(',')]

        # Create a TF-IDF matrix based on overview and genres
        df_copy = df.copy()
        df_copy['combined'] = df_copy['overview'].fillna('') + ' ' + df_copy['genres'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_copy['combined'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Get index of selected movie
        idx = df_copy.index[df_copy['normalized_title'] == movie_title.lower()].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # Exclude itself

        # Get recommended movies indices and filter by feedback
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = df_copy.iloc[movie_indices]
        recommended_movies = filter_low_feedback(recommended_movies, movie_title)

        # If low-rated movies were removed, replace them with new recommendations
        if len(recommended_movies) < 10:
            additional_recs = df_copy.loc[~df_copy.index.isin(recommended_movies.index) & ~df_copy['title'].eq(movie_title)]
            additional_recs = additional_recs.head(10 - len(recommended_movies))
            recommended_movies = pd.concat([recommended_movies, additional_recs])

        # Display recommendations
        st.subheader("Recommended Movies:")
        feedback_scores = {}

        for index, row in recommended_movies.head(10).iterrows():
            st.write(f"**Title:** {row['title']}")
            st.write(f"**Overview:** {row['overview']}")
            st.image(row['poster_path'], width=200)
            score = st.slider(f"Rate the recommendation for '{row['title']}'", 1, 10, value=5, key=f"slider_{row['title']}")
            feedback_scores[row['title']] = score

        if st.button("Submit Feedback"):
            log_feedback(movie_title, feedback_scores)
            st.success("Feedback submitted successfully. Future searches will consider this feedback.")
