# import required libraries
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer # count vectorizer
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
twentyColorPalette = sns.color_palette('hls', 20)
import plotly.offline as py
py.init_notebook_mode(connected=False)
# Classification model for positive and negative reviews
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn import svm # support verctor machine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Use English stop words
stop_words = stopwords.words('english')
# Use English stemmer.
stemmer = SnowballStemmer("english")
#funciton to split words into tokens
def identify_tokens(row):
    review = row['reviews.text']
    tokens = nltk.word_tokenize(review)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
# remove stop words from tokens 
# A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore)
def remove_stops(row):
    my_list = row['reviews.text']
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)
#stemming of words 
# Stemming is the process of reducing inflection in words to their root forms
def stem_list(row):
    my_list = row['reviews.text']
    stemmed_list = [stemmer.stem(word) for word in my_list]
    return (stemmed_list)
# remove punctuations that doesn't have any meaning
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return final

# Load hotel reviews dataset 
# Data has been collected from (https://www.kaggle.com/datafiniti/hotel-reviews?select=7282_1.csv)
hotel_review = pd.read_csv('7282_1.csv')
hotel_review_copy = hotel_review.copy()
# List hotels with highest reviews
print(hotel_review_copy['name'].value_counts()[0:3])
# Top 2 hotels used for review and sentiment analysis are
# 1. The Alexandrian, Autograph Collection
# 2. Howard Johnson Inn - Newburgh
#########################################################################################################################
# 1. The Alexandrian, Autograph Collection hotel review and sentiment analysis###########################################
#########################################################################################################################
hotel_reivew_1 = hotel_review_copy.loc[hotel_review['name']=='The Alexandrian, Autograph Collection']
hotel_reivew_1 = hotel_reivew_1[['name', 'reviews.date','reviews.text']]
hotel_reivew_1['reviews.text'] = hotel_reivew_1['reviews.text'].str.lower()
#tokenization
hotel_reivew_1['reviews.text'] = hotel_reivew_1.apply(identify_tokens, axis=1)
#get list of all hotels
hotel_list = list(hotel_review['name'].unique())

################################################################################################################
##########1.1. Number of reviews  of  "The Alexandrian, Autograph Collection" for period of time################
################################################################################################################
hotel_reivew_1['reviews.date'] = hotel_reivew_1['reviews.date'] .str[:7]
number_of_reviews_by_date = hotel_reivew_1.groupby(['reviews.date','name']).size().reset_index(name='counts').set_index(['reviews.date'])
sns.set()
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Number of reviews for period of time", fontsize=20)
plt.plot(number_of_reviews_by_date["counts"],linewidth=3.0)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14, rotation=90)
plt.xlabel("Time Period", fontsize=18)
plt.ylabel("Number of Reviews", fontsize=18)
plt.show()

################################################################################################################
##########################1.2. Unigrams(Most common words in review text(stop words removed))###################
################################################################################################################
hotel_reivew_1['reviews.text'] = hotel_reivew_1.apply(remove_stops, axis=1)
sns.set()
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Most common words in review text(stop words removed)", fontsize=20)
pd.Series(' '.join(' '.join(word) for word in hotel_reivew_1["reviews.text"])
             .split()).value_counts()[:25].plot.barh(color=twentyColorPalette)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.xlabel("Number of Users", fontsize=18)
plt.ylabel("Word", fontsize=18)
plt.show()

################################################################################################################
#####################1.3. Unigrams (Most common words in review text(stop words removed and stemmed))###########
################################################################################################################
hotel_reivew_1['reviews.text'] = hotel_reivew_1.apply(stem_list, axis=1)
sns.set()
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Most common words in review text(stop words removed and stemmed)", fontsize=20)
pd.Series(' '.join(' '.join(word) for word in hotel_reivew_1["reviews.text"])
             .split()).value_counts()[:25].plot.barh(color=twentyColorPalette)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.xlabel("Number of Users", fontsize=18)
plt.ylabel("Word", fontsize=18)
plt.show()

################################################################################################################
##########################1.4. Bigrams############################################################################
################################################################################################################
bigrams = list(ngrams(' '.join(' '.join(word) for word in hotel_reivew_1["reviews.text"])
             .split(),2))
counter_bigrams = Counter(bigrams)
bigram_df = pd.DataFrame.from_dict(counter_bigrams,orient='index').reset_index().nlargest(10,0)
################################################################################################################
##########################1.5. Trigrams###########################################################################
################################################################################################################
trigrams = list(ngrams(' '.join(' '.join(word) for word in hotel_reivew_1["reviews.text"])
             .split(),3))
counter_trigrams = Counter(trigrams)
trigram_df = pd.DataFrame.from_dict(counter_trigrams,orient='index').reset_index().nlargest(10,0)
################################################################################################################
##########################1.6. Sentiment Analysis of The Alexandrian, Autograph Collection reviews##############
################################################################################################################
hotel_reivew_1 = hotel_review_copy.loc[hotel_review['name']=='The Alexandrian, Autograph Collection']
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Most common words in review text(stop words removed)", fontsize=20)
plt.hist(x=hotel_reivew_1['reviews.rating'])
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.xlabel("Number of Users", fontsize=18)
plt.ylabel("Word", fontsize=18)
plt.show()

#################################################################################################################
##########################1.6.1. Positive and Negative sentiment words#############################################
#################################################################################################################
# Create stopword list:
type(stop_words)
textt = " ".join(review for review in hotel_reivew_1['reviews.text'])
wordcloud = WordCloud(stopwords=stop_words).generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#hotel_reivew_Hotel_Russo_Palace = hotel_reivew_Hotel_Russo_Palace[hotel_reivew_Hotel_Russo_Palace['reviews.rating'] != 3]
hotel_reivew_1['sentiment'] = hotel_reivew_1['reviews.rating'].apply(lambda rating : +1 if rating > 3 else -1)

# split df - positive and negative sentiment:
positive = hotel_reivew_1[hotel_reivew_1['sentiment'] == 1]
negative = hotel_reivew_1[hotel_reivew_1['sentiment'] == -1]
# Wordcloud — Positive Sentiment
pos = " ".join(review for review in positive['reviews.text'])
wordcloud2 = WordCloud(stopwords=stop_words).generate(pos)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.show()

# Wordcloud — Negative Sentiment
neg = " ".join(review for review in negative['reviews.text'])
wordcloud3 = WordCloud(stopwords=stop_words).generate(neg)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off") 
plt.show()

# Hotel Reviews Sentiment
hotel_reivew_1['sentiment'] = hotel_reivew_1['sentiment'].replace({-1 : 'negative'})
hotel_reivew_1['sentiment'] = hotel_reivew_1['sentiment'].replace({1 : 'positive'})
plt.hist(x=hotel_reivew_1['sentiment'])
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.xlabel("Number of Users", fontsize=18)
plt.ylabel("Word", fontsize=18)
plt.show()
#######################################################################################################################
# 1.7. Building the classificaiotn model to classify reviews between positive and negative ############################
#######################################################################################################################
# Data Cleaning
hotel_reivew_1['reviews.text'] = hotel_reivew_1['reviews.text'].apply(remove_punctuation)
hotel_reivew_1 = hotel_reivew_1.dropna(subset=['reviews.text'])
# Split the Dataframe
hotel_reivew_1_new = hotel_reivew_1[['reviews.text','sentiment']]
hotel_reivew_1_new.head()
# random split train and test data
index = hotel_reivew_1_new.index
hotel_reivew_1_new['random_number'] = np.random.randn(len(index))
train = hotel_reivew_1_new[hotel_reivew_1_new['random_number'] <= 0.75]
test = hotel_reivew_1_new[hotel_reivew_1_new['random_number'] > 0.75]
#Create a bag of words
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['reviews.text'])
test_matrix = vectorizer.transform(test['reviews.text'])
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']
##========================================================================================================#
# Build Logistic Regression
lr = LogisticRegression()
# fit model
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)
# testing with test dataset
print("Logistic Regression Classification report")
print(classification_report(predictions,y_test))
print("Accuracry Score")
print(accuracy_score(predictions,y_test))
data = {'y_Actual':    y_test,
        'y_Predicted': predictions
        }
# Display confusion matrix
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusionmatrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
plt.title("Confusion matrix (Linear Regression)", fontsize=20)
sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='gist_stern_r')
plt.xlabel("Predicted", size=18)
plt.ylabel("Actual", size=18)
plt.show()
#=========================================================================================================#
# Build Gaussian Naive Bayes 
gnb = GaussianNB()
# fit model
gnb.fit(X_train.toarray(),y_train)
predictions = gnb.predict(X_test.toarray())
# testing with test dataset
print("Gaussian Naive Bayes Classification report")
print(classification_report(predictions,y_test))
print("Accuracry Score")
print(accuracy_score(predictions,y_test))
data = {'y_Actual':    y_test,
        'y_Predicted': predictions
        }
# Display confusion matrix
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusionmatrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
plt.title("Confusion matrix (Gaussian Naive Bayes)", fontsize=20)
sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='gist_stern_r')
plt.xlabel("Predicted", size=18)
plt.ylabel("Actual", size=18)
plt.show()
#=========================================================================================================#
# Build Support Vector Machines Classifier
svmc = svm.SVC()
# fit model
svmc.fit(X_train,y_train)
predictions = svmc.predict(X_test)
# testing with test dataset
print("Support Vector Machines Classification report")
print(classification_report(predictions,y_test))
print("Accuracry Score")
print(accuracy_score(predictions,y_test))
data = {'y_Actual':    y_test,
        'y_Predicted': predictions
        }
# Display confusion matrix
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusionmatrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
plt.title("Confusion matrix (Support Vector Machines Classifier)", fontsize=20)
sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='gist_stern_r')
plt.xlabel("Predicted", size=18)
plt.ylabel("Actual", size=18)
plt.show()
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
# 2. Howard Johnson Inn - Newburgh hotel review and sentiment analysis###########################################
#########################################################################################################################
hotel_reivew_2 = hotel_review_copy.loc[hotel_review['name']=='Howard Johnson Inn - Newburgh']
hotel_reivew_2 = hotel_reivew_2[['name', 'reviews.date','reviews.text']]
hotel_reivew_2['reviews.text'] = hotel_reivew_2['reviews.text'].str.lower()
#tokenization
hotel_reivew_2['reviews.text'] = hotel_reivew_2.apply(identify_tokens, axis=1)

################################################################################################################
##########2.1. Number of reviews  of  "Howard Johnson Inn - Newburgh" for period of time################
################################################################################################################
hotel_reivew_2['reviews.date'] = hotel_reivew_2['reviews.date'] .str[:7]
number_of_reviews_by_date = hotel_reivew_2.groupby(['reviews.date','name']).size().reset_index(name='counts').set_index(['reviews.date'])
sns.set()
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Number of reviews for period of time", fontsize=20)
plt.plot(number_of_reviews_by_date["counts"],linewidth=3.0)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14, rotation=90)
plt.xlabel("Time Period", fontsize=18)
plt.ylabel("Number of Reviews", fontsize=18)
plt.show()

################################################################################################################
##########################2.2. Unigrams(Most common words in review text(stop words removed))###################
################################################################################################################
hotel_reivew_2['reviews.text'] = hotel_reivew_2.apply(remove_stops, axis=1)
sns.set()
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Most common words in review text(stop words removed)", fontsize=20)
pd.Series(' '.join(' '.join(word) for word in hotel_reivew_2["reviews.text"])
             .split()).value_counts()[:25].plot.barh(color=twentyColorPalette)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.xlabel("Number of Users", fontsize=18)
plt.ylabel("Word", fontsize=18)
plt.show()

################################################################################################################
#####################2.3. Unigrams (Most common words in review text(stop words removed and stemmed))###########
################################################################################################################
hotel_reivew_2['reviews.text'] = hotel_reivew_2.apply(stem_list, axis=1)
sns.set()
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Most common words in review text(stop words removed and stemmed)", fontsize=20)
pd.Series(' '.join(' '.join(word) for word in hotel_reivew_2["reviews.text"])
             .split()).value_counts()[:25].plot.barh(color=twentyColorPalette)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.xlabel("Number of Users", fontsize=18)
plt.ylabel("Word", fontsize=18)
plt.show()

################################################################################################################
##########################2.4. Bigrams############################################################################
################################################################################################################
bigrams = list(ngrams(' '.join(' '.join(word) for word in hotel_reivew_2["reviews.text"])
             .split(),2))
counter_bigrams = Counter(bigrams)
bigram_df = pd.DataFrame.from_dict(counter_bigrams,orient='index').reset_index().nlargest(10,0)
################################################################################################################
##########################1.5. Trigrams###########################################################################
################################################################################################################
trigrams = list(ngrams(' '.join(' '.join(word) for word in hotel_reivew_2["reviews.text"])
             .split(),3))
counter_trigrams = Counter(trigrams)
trigram_df = pd.DataFrame.from_dict(counter_trigrams,orient='index').reset_index().nlargest(10,0)

################################################################################################################
##########################2.6. Sentiment Analysis of Howard Johnson Inn - Newburgh reviews##############
################################################################################################################
hotel_reivew_2 = hotel_review_copy.loc[hotel_review['name']=='Howard Johnson Inn - Newburgh']
plt.rcParams["figure.figsize"] = (20,10)
plt.title("Most common words in review text(stop words removed)", fontsize=20)
plt.hist(x=hotel_reivew_2['reviews.rating'])
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.xlabel("Number of Users", fontsize=18)
plt.ylabel("Word", fontsize=18)
plt.show()

#################################################################################################################
##########################2.6.1. Positive and Negative sentiment words#############################################
#################################################################################################################
# Create stopword list:
type(stop_words)
textt = " ".join(review for review in hotel_reivew_2['reviews.text'])
wordcloud = WordCloud(stopwords=stop_words).generate(textt)
plt.title("Wordcloud of positive and negative sentiment words", fontsize=20)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#hotel_reivew_Hotel_Russo_Palace = hotel_reivew_Hotel_Russo_Palace[hotel_reivew_Hotel_Russo_Palace['reviews.rating'] != 3]
hotel_reivew_2['sentiment'] = hotel_reivew_2['reviews.rating'].apply(lambda rating : +1 if rating > 3 else -1)

# split df - positive and negative sentiment:
positive = hotel_reivew_2[hotel_reivew_2['sentiment'] == 1]
negative = hotel_reivew_2[hotel_reivew_2['sentiment'] == -1]
# Wordcloud — Positive Sentiment
pos = " ".join(review for review in positive['reviews.text'])
wordcloud2 = WordCloud(stopwords=stop_words).generate(pos)
plt.title("Wordcloud of positive sentiment words", fontsize=20)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.show()

# Wordcloud — Negative Sentiment
neg = " ".join(review for review in negative['reviews.text'])
wordcloud3 = WordCloud(stopwords=stop_words).generate(neg)
plt.title("Wordcloud of negative sentiment words", fontsize=20)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off") 
plt.show()

# Hotel Reviews Sentiment
hotel_reivew_2['sentiment'] = hotel_reivew_2['sentiment'].replace({-1 : 'negative'})
hotel_reivew_2['sentiment'] = hotel_reivew_2['sentiment'].replace({1 : 'positive'})
plt.hist(x=hotel_reivew_2['sentiment'])
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.xlabel("Number of Users", fontsize=18)
plt.ylabel("Word", fontsize=18)
plt.show()
########################################################################################################################################
# 2.7. Building the classificaiotn model to classify reviews between positive and negative(Howard Johnson Inn - Newburgh)###############
########################################################################################################################################
# Data Cleaning
hotel_reivew_2['reviews.text'] = hotel_reivew_2['reviews.text'].apply(remove_punctuation)
hotel_reivew_2 = hotel_reivew_2.dropna(subset=['reviews.text'])
# Split the Dataframe
hotel_reivew_2_new = hotel_reivew_2[['reviews.text','sentiment']]
hotel_reivew_2_new.head()
# random split train and test data
index = hotel_reivew_2_new.index
hotel_reivew_2_new['random_number'] = np.random.randn(len(index))
train = hotel_reivew_2_new[hotel_reivew_2_new['random_number'] <= 0.75]
test = hotel_reivew_2_new[hotel_reivew_2_new['random_number'] > 0.75]
#Create a bag of words
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['reviews.text'])
test_matrix = vectorizer.transform(test['reviews.text'])
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']
#========================================================================================================#
# Build Logistic Regression
lr = LogisticRegression()
# fit model
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)
# testing with test dataset
print("Logistic Regression Classification report")
print(classification_report(predictions,y_test))
print("Accuracry Score")
print(accuracy_score(predictions,y_test))
data = {'y_Actual':    y_test,
        'y_Predicted': predictions
        }
# Display confusion matrix
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusionmatrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
plt.title("Confusion matrix (Linear Regression)", fontsize=20)
sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='gist_stern_r')
plt.xlabel("Predicted", size=18)
plt.ylabel("Actual", size=18)
plt.show()
#=========================================================================================================#
# Build Gaussian Naive Bayes 
gnb = GaussianNB()

# fit model
gnb.fit(X_train.toarray(),y_train)
predictions = gnb.predict(X_test.toarray())
# testing with test dataset
print("Gaussian Naive Bayes Classification report")
print(classification_report(predictions,y_test))
print("Accuracry Score")
print(accuracy_score(predictions,y_test))
data = {'y_Actual':    y_test,
        'y_Predicted': predictions
        }
# Display confusion matrix
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusionmatrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
plt.title("Confusion matrix (Gaussian Naive Bayes)", fontsize=20)
sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='gist_stern_r')
plt.xlabel("Predicted", size=18)
plt.ylabel("Actual", size=18)
plt.show()
#=========================================================================================================#
# Build Support Vector Machines Classifier
svmc = svm.SVC()
# fit model
svmc.fit(X_train,y_train)
predictions = svmc.predict(X_test)
# testing with test dataset
print("Support Vector Machines Classification report")
print(classification_report(predictions,y_test))
print("Accuracry Score")
print(accuracy_score(predictions,y_test))
data = {'y_Actual':    y_test,
        'y_Predicted': predictions
        }
# Display confusion matrix
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusionmatrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
plt.title("Confusion matrix (Support Vector Machines Classifier)", fontsize=20)
sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='gist_stern_r')
plt.xlabel("Predicted", size=18)
plt.ylabel("Actual", size=18)
plt.show()
#######################################################################################################################
####################################3. All Hotels######################################################################
#######################################################################################################################
# 3.1. Building the classificaiotn model to classify reviews between positive and negative for all hotels ###############
#######################################################################################################################
hotel_review_all = hotel_review_copy.copy()
hotel_review_all['sentiment'] = hotel_review_all['reviews.rating'].apply(lambda rating : +1 if rating > 3 else -1)
# Data Cleaning
hotel_review_all = hotel_review_all.dropna(subset=['reviews.text'])
# Split the Dataframe
hotel_review_all = hotel_review_all[['reviews.text','sentiment']]
hotel_review_all.head()
# random split train and test data
index = hotel_review_all.index
hotel_review_all['random_number'] = np.random.randn(len(index))
train = hotel_review_all[hotel_review_all['random_number'] <= 0.75]
test = hotel_review_all[hotel_review_all['random_number'] > 0.75]
#Create a bag of words
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['reviews.text'])
test_matrix = vectorizer.transform(test['reviews.text'])
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']
#========================================================================================================#
# Build Logistic Regression
lr = LogisticRegression()
# fit model
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)
# testing with test dataset
print("Logistic Regression Classification report")
print(classification_report(predictions,y_test))
print("Accuracry Score")
print(accuracy_score(predictions,y_test))
data = {'y_Actual':    y_test,
        'y_Predicted': predictions
        }
# Display confusion matrix
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusionmatrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
plt.title("Confusion matrix (Linear Regression)", fontsize=20)
sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='gist_stern_r')
plt.xlabel("Predicted", size=18)
plt.ylabel("Actual", size=18)
plt.show()
#=========================================================================================================#
# Build Gaussian Naive Bayes 
gnb = GaussianNB()

# fit model
gnb.fit(X_train.toarray(),y_train)
predictions = gnb.predict(X_test.toarray())
# testing with test dataset
print("Gaussian Naive Bayes Classification report")
print(classification_report(predictions,y_test))
print("Accuracry Score")
print(accuracy_score(predictions,y_test))
data = {'y_Actual':    y_test,
        'y_Predicted': predictions
        }
# Display confusion matrix
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusionmatrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
plt.title("Confusion matrix (Gaussian Naive Bayes)", fontsize=20)
sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='gist_stern_r')
plt.xlabel("Predicted", size=18)
plt.ylabel("Actual", size=18)
plt.show()
#=========================================================================================================#
# Build Support Vector Machines Classifier
svmc = svm.SVC()
# fit model
svmc.fit(X_train,y_train)
predictions = svmc.predict(X_test)
# testing with test dataset
print("Support Vector Machines Classification report")
print(classification_report(predictions,y_test))
print("Accuracry Score")
print(accuracy_score(predictions,y_test))
data = {'y_Actual':    y_test,
        'y_Predicted': predictions
        }
# Display confusion matrix
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusionmatrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
plt.title("Confusion matrix (Support Vector Machines Classifier)", fontsize=20)
sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='gist_stern_r')
plt.xlabel("Predicted", size=18)
plt.ylabel("Actual", size=18)
plt.show()
############################################################################################################
############################################################################################################
############################################################################################################


