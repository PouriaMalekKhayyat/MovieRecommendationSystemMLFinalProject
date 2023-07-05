import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import requests

movies_raw = pd.read_csv('dataset/movies_raw.csv')
smd = pd.read_csv('dataset/smd.csv')
smd.fillna('', inplace=True)
indices = pd.Series(smd.index, index=smd['title'])
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=020b311fe0559698373a16008dc6a672&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

vote_counts = movies_raw[movies_raw['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies_raw[movies_raw['vote_average'].notnull()]['vote_average'].astype('int')

C = vote_averages.mean()
m = vote_counts.quantile(0.95)

def add_line_break(string):
    words = string.split()
    new_string = ""
    for i, word in enumerate(words):
        new_string += word
        if (i + 1) % 4 == 0:
            new_string += "\n"
        else:
            new_string += " "
    return new_string

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'id', 'vote_average', 'vote_count', 'overview']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')

    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)

    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified = qualified.head(5)

    recommended_movies_posters = []
    movie_id_list = qualified['id'].tolist()
    for movie_id in movie_id_list:
        recommended_movies_posters.append(fetch_poster(movie_id))
    
    qualified['vote_average'] = qualified['vote_average'].astype(str)
    qualified['overview'] = qualified['overview'].apply(add_line_break)

    return qualified['title'].tolist(), recommended_movies_posters, qualified['vote_average'].tolist(), qualified['overview'].tolist()

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Select Your favorite Movie!',
    movies_raw['title'].values
)

if st.button('Recommend'):
    names, posters, votes, overviews = improved_recommendations(selected_movie_name)
    col1, col2= st.columns(2, gap='large')
    with col1:
        st.text(names[0])
        st.image(posters[0], width=100)
        st.text('score: ' + votes[0])
        st.text(names[1])
        st.image(posters[1], width=100)
        st.text('score: ' + votes[1])
        st.text(names[2])
        st.image(posters[2], width=100)
        st.text('score: ' + votes[2])
        st.text(names[3])
        st.image(posters[3], width=100)
        st.text('score: ' + votes[3])
        st.text(names[4])
        st.image(posters[4], width=100)
        st.text('score: ' + votes[4])
    with col2:
        st.text(names[0] + ': ')
        st.text(overviews[0])
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.text(names[1] + ': ')
        st.text(overviews[1])
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.text(names[2] + ': ')
        st.text(overviews[2])
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.text(names[3] + ': ')
        st.text(overviews[3])
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.text(names[4] + ': ')
        st.text(overviews[4])
