import pandas as pd
from textblob import TextBlob

def add_sentiment_column(df):
    """Adds a sentiment score column to the dataframe."""
    df['sentiment'] = df['overview'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notnull(x) else 0)
    return df
 