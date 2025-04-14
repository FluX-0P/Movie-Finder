import pandas as pd

# Load your dataset
df = pd.read_csv('tmdb_movies.csv')

# Display the original number of rows
original_count = df.shape[0]
print(f"Original number of movies: {original_count}")

# Step 1: Filter out adult movies
df_filtered = df[df['adult'] == False]  # Keep only non-adult films

# Step 2: Additional filtering based on keywords in title and genres
# Define a list of keywords to exclude
keywords_to_exclude = ['sex', 'porn', 'adult', 'erotic', 'nude', 'explicit']

# Filter out movies with those keywords in the title or genres
df_filtered = df_filtered[
    ~df_filtered['title'].str.contains('|'.join(keywords_to_exclude), case=False, na=False) &
    ~df_filtered['genres'].str.contains('|'.join(keywords_to_exclude), case=False, na=False)
]

# Step 3: Convert release_date to datetime format for filtering
df_filtered['release_date'] = pd.to_datetime(df_filtered['release_date'], errors='coerce')

# Step 4: Filter for Hindi and English movies made after 1970
year_threshold = 1970
df_filtered = df_filtered[
    (df_filtered['release_date'].dt.year > year_threshold) &
    (df_filtered['original_language'].isin(['en', 'hi']))  # Only Hindi and English movies
]

# Step 5: Filter for popular movies
popularity_threshold = 10  # Adjust this threshold based on your definition of "popular"
df_filtered = df_filtered[df_filtered['popularity'] > popularity_threshold]

# Display the number of rows after filtering
filtered_count = df_filtered.shape[0]
print(f"Number of movies after filtering for non-adult Hindi and English films made after 1970 and popular: {filtered_count}")

# Optional: Sample 10,000 films if there are at least 10,000 available
# if filtered_count >= 10000:
#     df_sampled = df_filtered.sample(n=10000, random_state=1)  # Randomly sample 10,000 films
# else:
#     df_sampled = df_filtered  # Use all available films if less than 10,000

# Display the number of rows after sampling
# sampled_count = df_sampled.shape[0]
# print(f"Number of movies after sampling: {sampled_count}")

# Optionally, save the sampled dataset
df_filtered.to_csv('filtered_tmdb_movies.csv', index=False)
