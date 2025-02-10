Netflix Recommendation System using Python

Netflix is a subscription-based streaming platform that allows users to watch movies and TV shows without advertisements. One of the reasons behind the popularity of Netflix is its recommendation system. Its recommendation system recommends movies and TV shows based on the user’s interest. If you are a Data Science student and want to learn how to create a Netflix recommendation system, this article is for you. This article will take you through how to build a Netflix recommendation system using Python.


Here’s How Netflix Recommendation System Works
The recommendation system of Netflix shows you movies and TV shows according to your interests. Netflix has a lot of data because of its user base. Its recommendation system predicts a personalised catalogue for you based on factors like:

your viewing history
the viewing history of other users with similar tastes and preferences as yours
genres, category, description, and more information about the content that you watched in the past
The genre of the content is one of the most valuable factors that helps Netflix recommend more content even to new users. I hope you have understood how Netflix recommends content to its users. You can learn more about it here. In the section below, I will take you through how to build a Netflix recommendation system using Python.

Netflix Recommendation System using Python
The dataset I am using to build a Netflix recommendation system using Python is downloaded from Kaggle. The dataset contains information about all the movies and TV shows on Netflix as of 2021. You can download the dataset from here.


Now let’s import the necessary Python libraries and the dataset we need for this task:

data = pd.read_csv("netflixData.csv")
print(data.head())
1
import numpy as np
2
import pandas as pd
3
from sklearn.feature_extraction import text
4
from sklearn.metrics.pairwise import cosine_similarity
5
​
6
data = pd.read_csv("netflixData.csv")
7
print(data.head())
                                Show Id                          Title  \
0  cc1b6ed9-cf9e-4057-8303-34577fb54477                       (Un)Well   
1  e2ef4e91-fb25-42ab-b485-be8e3b23dedb                         #Alive   
2  b01b73b7-81f6-47a7-86d8-acb63080d525  #AnneFrank - Parallel Stories   
3  b6611af0-f53c-4a08-9ffa-9716dc57eb9c                       #blackAF   
4  7f2d4170-bab8-4d75-adc2-197f7124c070               #cats_the_mewvie   

                                         Description  \
0  This docuseries takes a deep dive into the luc...   
1  As a grisly virus rampages a city, a lone man ...   
2  Through her diary, Anne Frank's story is retol...   
3  Kenya Barris and his family navigate relations...   
4  This pawesome documentary explores how our fel...   

                      Director  \
0                          NaN   
1                       Cho Il   
2  Sabina Fedeli, Anna Migotto   
3                          NaN   
4             Michael Margolis   

                                           Genres  \
0                                      Reality TV   
1  Horror Movies, International Movies, Thrillers   
2             Documentaries, International Movies   
3                                     TV Comedies   
4             Documentaries, International Movies   

                                                Cast Production Country  \
0                                                NaN      United States   
1                           Yoo Ah-in, Park Shin-hye        South Korea   
2                        Helen Mirren, Gengher Gatti              Italy   
3  Kenya Barris, Rashida Jones, Iman Benson, Genn...      United States   
4                                                NaN             Canada   

   Release Date Rating  Duration Imdb Score Content Type         Date Added  
0        2020.0  TV-MA  1 Season     6.6/10      TV Show                NaN  
1        2020.0  TV-MA    99 min     6.2/10        Movie  September 8, 2020  
2        2019.0  TV-14    95 min     6.4/10        Movie       July 1, 2020  
3        2020.0  TV-MA  1 Season     6.6/10      TV Show                NaN  
4        2020.0  TV-14    90 min     5.1/10        Movie   February 5, 2020  
In the first impressions on the dataset, I can see that the Title column needs preparation as it contains # before the name of the movies or tv shows. I will get back to it. For now, let’s have a look at whether the data contains null values or not:

print(data.isnull().sum())
1
print(data.isnull().sum())
Show Id                  0
Title                    0
Description              0
Director              2064
Genres                   0
Cast                   530
Production Country     559
Release Date             3
Rating                   4
Duration                 3
Imdb Score             608
Content Type             0
Date Added            1335
dtype: int64
The dataset contains null values, but before removing the null values, let’s select the columns that we can use to build a Netflix recommendation system:

1
data = data[["Title", "Description", "Content Type", "Genres"]]
2
print(data.head())
                           Title  \
0                       (Un)Well   
1                         #Alive   
2  #AnneFrank - Parallel Stories   
3                       #blackAF   
4               #cats_the_mewvie   

                                         Description Content Type  \
0  This docuseries takes a deep dive into the luc...      TV Show   
1  As a grisly virus rampages a city, a lone man ...        Movie   
2  Through her diary, Anne Frank's story is retol...        Movie   
3  Kenya Barris and his family navigate relations...      TV Show   
4  This pawesome documentary explores how our fel...        Movie   

                                           Genres  
0                                      Reality TV  
1  Horror Movies, International Movies, Thrillers  
2             Documentaries, International Movies  
3                                     TV Comedies  
4             Documentaries, International Movies  
As the name suggests:


The title column contains the titles of movies and TV shows on Netflix
Description column describes the plot of the TV shows and movies
The Content Type column tells us if it’s a movie or a TV show
The Genre column contains all the genres of the TV show or the movie
Now let’s drop the rows containing null values and move further:

data = data.dropna()
1
data = data.dropna()
Now I will clean the Title column as it contains some data preparation:

1
import nltk
2
import re
3
nltk.download('stopwords')
4
stemmer = nltk.SnowballStemmer("english")
5
from nltk.corpus import stopwords
6
import string
7
stopword=set(stopwords.words('english'))
8
​
9
def clean(text):
10
    text = str(text).lower()
11
    text = re.sub('\[.*?\]', '', text)
12
    text = re.sub('https?://\S+|www\.\S+', '', text)
13
    text = re.sub('<.*?>+', '', text)
14
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
15
    text = re.sub('\n', '', text)
16
    text = re.sub('\w*\d\w*', '', text)
17
    text = [word for word in text.split(' ') if word not in stopword]
18
    text=" ".join(text)
19
    text = [stemmer.stem(word) for word in text.split(' ')]
20
    text=" ".join(text)
21
    return text
22
data["Title"] = data["Title"].apply(clean)
Now let’s have a look at some samples of the Titles before moving forward:

print(data.Title.sample(10))
1
print(data.Title.sample(10))
3111           miniforc super dino power
1822                         girl reveng
910                        casino tycoon
4075                          sand castl
2760                                lock
3406                          nightflyer
536     bangkok love stori object affect
4365                             special
1733                                full
2343                     jeff dunham map
Name: Title, dtype: object
Now I will use the Genres column as the feature to recommend similar content to the user. I will use the concept of cosine similarity here (used to find similarities in two documents):

1
feature = data["Genres"].tolist()
2
tfidf = text.TfidfVectorizer(input=feature, stop_words="english")
3
tfidf_matrix = tfidf.fit_transform(feature)
4
similarity = cosine_similarity(tfidf_matrix)
Now I will set the Title column as an index so that we can find similar content by giving the title of the movie or TV show as an input:


1
indices = pd.Series(data.index, 
2
                    index=data['Title']).drop_duplicates()
Now here’s how to write a function to recommend Movies and TV shows on Netflix:

def netFlix_recommendation(title, similarity = similarity):
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    movieindices = [i[0] for i in similarity_scores]
    return data['Title'].iloc[movieindices]

print(netFlix_recommendation("girlfriend"))
1
def netFlix_recommendation(title, similarity = similarity):
2
    index = indices[title]
3
    similarity_scores = list(enumerate(similarity[index]))
4
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
5
    similarity_scores = similarity_scores[0:10]
6
    movieindices = [i[0] for i in similarity_scores]
7
    return data['Title'].iloc[movieindices]
8
​
9
print(netFlix_recommendation("girlfriend"))
3                          blackaf
285                     washington
417                 arrest develop
434     astronomi club sketch show
451    aunti donna big ol hous fun
656                      big mouth
752                bojack horseman
805                   brew brother
935                       champion
937                  chappell show
Name: Title, dtype: object
So this is how you can build a Netflix Recommendation System using the Python programming language.
