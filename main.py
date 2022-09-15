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

from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

#%matplotlib inline

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
	df = pd.read_csv(r"D:\Open Classroom\Datasets\Netflix\netflix_titles.csv")
	
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

	tf_vect = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1,3), 
                          stop_words='english', strip_accents='unicode')

	df['description'] = df['description'].fillna('')

	desc_matric = tf_vect.fit_transform(df['description'])

	st.text('we peek through the TF-IDF shape')
	st.write(desc_matric.shape)

	st.write('Since we have converted the series to a matrix, we use the Sigmoid Kernel to compute the metrics pairwise of the content and prediction (X, Y)')
	sig = sigmoid_kernel(desc_matric, desc_matric)

	# Create a new dataframe holding the movie titles and index series
	indices = pd.Series(df.index, df['title']).drop_duplicates()

	def netflix_recommender(title, sig=sig):
    
	    db_mov_list = [x for x in df['title']]
	    
	    if title in db_mov_list:
	    
	        idx = indices[title]

	        mov_list = list(enumerate(sig[idx]))

	        sort_mov = sorted(mov_list, key = lambda x: x[1], reverse = True)

	        top_ten = sort_mov[0:10]

	        movies_rec = [x[0] for x in top_ten]
	        
	        st.text("")
	        
	        st.text('Heres a list of Top 10 Movies we recommend you to watch')

	        return st.write(df[['title', 'type', 'listed_in', 'duration', 'release_year']].iloc[movies_rec].sort_values(by = ["release_year", "duration"], ascending = False))
	    
	    else:
	        st.text('__DATABASE ERROR___ During handling of the above exception, another exception occurred: InvalidIndexError(key)')
	        st.text('MOVIE NOT FOUND')


	st.image("Build-a-Recommendation-Engine-With-Collaborative-Filtering_Watermarked.webp")
	
	netflix_recommender(select_movie)
	
	with st.spinner("Loading..."):
		time.sleep(5)
		st.success("Done!")


	st.markdown('credits @ntonie')


