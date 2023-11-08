# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load your ratings dataset (replace 'ratings.csv' with your actual dataset file)
df_ratings = pd.read_csv('Datasets/ratings.csv')

# Load your movies dataset (replace 'movies.csv' with your actual dataset file)
df_movies = pd.read_csv('Datasets/movies.csv')

# Merge ratings and movies dataframes
df = pd.merge(df_ratings, df_movies, left_on='movie_id', right_on='movie_id')

# Create a user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Set up Flask
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    
    # Get user's ratings and movies they haven't rated
    user_ratings = user_item_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings == 0].index

    # Calculate average similarity with other users
    user_avg_similarity = user_similarity[user_id - 1].mean()

    # Calculate weighted average of ratings for unseen movies
    user_sim_vector = user_similarity[user_id - 1][:, None]  # Convert to column vector
    weighted_avg = (user_sim_vector * user_item_matrix.loc[:, unseen_movies].values).sum(axis=0) / user_avg_similarity

    # Sort the recommendations based on weighted average ratings
    recommendations_df = pd.DataFrame({'movie_id': unseen_movies, 'weighted_avg': weighted_avg})
    recommendations_df = recommendations_df.sort_values(by='weighted_avg', ascending=False).head(10)

    # Merge with movies dataframe to get movie names and ratings
    recommended_movies = pd.merge(recommendations_df, df_movies, on='movie_id')[['title', 'weighted_avg']]
    recommended_movies.columns = ['movie_name', 'rating']

    # Print recommendations once
    print(f"Top 10 Recommended Movies for User {user_id}")
    for idx, row in recommended_movies.iterrows():
        print(f"{row['movie_name']} - Rating: {row['rating']}")

    return render_template('recommendations.html', user_id=user_id, movies=recommended_movies.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
