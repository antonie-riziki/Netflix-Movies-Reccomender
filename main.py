import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import missingno as msno
import autoreload
import warnings
import os
import csv
import sys
import time
import requests

from collections import defaultdict, Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# %matplotlib inline

sb.set()
sb.set_style('darkgrid')
sb.set_palette('viridis')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

warnings.filterwarnings('ignore')


# instances of the application

header = st.container()
data_loading = st.container()
feature_eng = st.container()
side_col = st.container()





# Header section

with header:
	st.title('Netflix Movies Recommendation System')
	st.image("Netflix-Continue-Watching.gif")
	st.write('While recommendation systems have gained the popularity in the advanced of internet usage, there has been need for developing systems that will however meet user preferences')
	st.write('On the other hand, we specified the choice of Content Based System rather than Colaborative Filtering')
	st.markdown('Lets roll our sleeves and get it done')



with data_loading:
	st.subheader('Data Loading and Exploration')
	df = pd.read_csv(r"netflix_titles.csv")
	
	
	st.write(df.head())

	with open("netflix_titles.csv", "rb") as file:
		st.download_button(label = 'download csv file', data = file, file_name = "netflix_titles.csv")
	
	st.write('The dataframe has ' + str(df.shape[0]) + ' features and ' + str(df.shape[1]) + ' records')

	st.text('Statistical Representation of the Dataset')
	col1, col2 = st.columns(2)

	
	with col1:
	
		st.text('Dataset Decsription')
		st.write(df.describe())

	with col2:
		st.text('Inpect missing values (NaNs')
		st.write(df.isnull().sum())

	st.text('Graphical Representation of Nan Values')
	st.bar_chart(df.isnull().sum())

	st.markdown('Since the dataset has NaN values, we will not clean since thats not the objective of this system, but for future accuracy of the model, we will have to.')



with st.sidebar:
    select_movie = st.selectbox('select a movie', (sorted(df['title'])))

with feature_eng:
	st.header('Feature Engineering')
	st.text('Content-Based Recommendation System')
	st.image("African-movies-on-Netflix.png")
	st.write('We will use the TF-IDF vectorizer to evaluate the "overview" series and convert it to a Document Term Matrix for evaluation')


	def clean_data(x):
	    return str.lower(x.replace(" ", ""))

	def create_soup(x):
	    return x['title']+ ' ' +  x['cast'] + ' ' + x['listed_in'] + ' ' + x['rating'] + ' ' + x['duration']

	def recommend_movie_general(title, cosine_sim):
	    global result
	    title=title.replace(' ','').lower()
	    idx = indices[title]
	    sim_scores = list(enumerate(cosine_sim[idx]))
	    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	    sim_scores = sim_scores[0:300]
	    movie_indices = [i[0] for i in sim_scores]
	    result =  df[['show_id','type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']].iloc[movie_indices]
	    result.reset_index(drop=True, inplace=True)
	    return result


	df = df.fillna('')

	new_features = ['title', 'cast', 'listed_in', 'rating', 'duration']
	netflix_data = df[new_features]

	for i in new_features:
	    netflix_data[i] = df[i].apply(clean_data)

	netflix_data['soup'] = netflix_data.apply(create_soup, axis=1)

	cv = CountVectorizer(stop_words='english')
	count_matrix = cv.fit_transform(netflix_data['soup'])

	# global cosine_sim

	cosine_sim = cosine_similarity(count_matrix, count_matrix)

	netflix_data=netflix_data.reset_index()

	indices = pd.Series(netflix_data.index, index=netflix_data['title'])
	# st.write('Heres the indices series')
	# st.dataframe(indices)

	recommend_movie_general(select_movie, cosine_sim)


	tf_vect = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1,3), 
                          stop_words='english', strip_accents='unicode')

	result['description'] = result['description'].fillna('')

	desc_matric = tf_vect.fit_transform(result['description'])

	st.text('we peek through the TF-IDF shape')
	st.write(desc_matric.shape)

	st.write('Since we have converted the series to a matrix, we use the Sigmoid Kernel to compute the metrics pairwise of the content and prediction (X, Y)')
	sig = sigmoid_kernel(desc_matric, desc_matric)

	# Create a new dataframe holding the movie titles and index series
	indices = pd.Series(result.index, result['title']).drop_duplicates()


	def fetch_movie_image(movie_title):
	    api_key = 'db89e680'
	    base_url = 'http://www.omdbapi.com/?apikey=' + api_key + '&t='
	    response = requests.get(base_url + movie_title)
	    movie_data = response.json()

	    if 'Poster' in movie_data:
	        return movie_data['Poster']
	    else:
	        return 'https://via.placeholder.com/200x300.png?text=No+Image+Available'
	    


	def recommend_movie_narrative(title, sig=sig):
    
	    db_mov_list = [x for x in result['title']]
	    
	    if title in db_mov_list:
	    
	        idx = indices[title]

	        mov_list = list(enumerate(sig[idx]))

	        sort_mov = sorted(mov_list, key = lambda x: x[1], reverse = True)

	        top_ten = sort_mov[1:21]

	        movies_rec = [x[0] for x in top_ten]
	        
	        st.text("")
	        
	        st.text('Your watchlist awaits')

	        # for i in range(0, len(movies_rec), 3):
	        #     row = movies_rec[i:i + 3]
	        #     with st.expander("", expanded=True):
	        #         st.image(fetch_movie_image(df['title'][row[0]]), width=200)
	        #         st.image(fetch_movie_image(df['title'][row[1]]), width=200)
	        #         st.image(fetch_movie_image(df['title'][row[2]]), width=200)
	        #     st.write("")

	        for i in range(0, len(movies_rec), 3):
	        	
	        	col1, col2, col3 = st.columns(3)
	        	row = movies_rec[i:i + 3]

	        	with col1:
	        		if len(row)>=1:
	        			st.image(fetch_movie_image(result['title'][row[0]]), caption=result['title'][row[0]], width=200)
	        	with col2:
	        		if len(row)>=2:
	        			st.image(fetch_movie_image(result['title'][row[1]]), caption=result['title'][row[1]], width=200)
	        	with col3:
	        		if len(row)>=3:
	        			st.image(fetch_movie_image(result['title'][row[2]]), caption=result['title'][row[2]], width=200)


	        # for i in movies_rec:
	        #     movie_title = df.iloc[i]['title']
	        #     movie_img_url = fetch_movie_image(movie_title)
	        #     st.image(movie_img_url, width=200)

	        # return st.write(df[['title', 'type', 'listed_in', 'duration', 'release_year']].iloc[movies_rec].sort_values(by = ["release_year", "duration"], ascending = False))
	    
	    else:
	        st.text('__DATABASE ERROR___ During handling of the above exception, another exception occurred: InvalidIndexError(key)')
	        st.text('MOVIE NOT FOUND')


	st.image("Build-a-Recommendation-Engine-With-Collaborative-Filtering_Watermarked.webp")
	
	recommend_movie_narrative(result['title'].iloc()[0])
	# netflix_recommender(select_movie)
	
	with st.spinner("Loading..."):
		time.sleep(5)
		st.success("Done!")


	st.markdown('credits EchoMinds Innovation ([EchoMinds Innovation])')
                                                                                                                                                                     
