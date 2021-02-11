# Natural-language-processing_YELP-reviews
PROJECT OVERVIEW: In this project,Natural language processing(NLP) strategies will be used to analyze Yelp reviews data. Yelp is an app that provides a crowd-sourced review forum to business and services. This app is used  to publish crowd-sourced reviews about businesses. Number of 'stars' indicate the business rating given by a customer, ranging from 1 to 5. 'Cool', 'Useful' and 'Funny' indicate the number of cool votes given by the other Yelp Users. CountVectorizer is used to achieve the success of the project '''

'''
PROJECT OVERVIEW:
In this project,Natural language processing(NLP) strategies will be used to analyze Yelp reviews data.
Yelp is an app that provides a crowd-sourced review forum to business and services. This app is used 
to publish crowd-sourced reviews about businesses.
Number of 'stars' indicate the business rating given by a customer, ranging from 1 to 5.
'Cool', 'Useful' and 'Funny' indicate the number of cool votes given by the other Yelp Users.
CountVectorizer is used to achieve the success of the project
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer

yelp_df = pd.read_csv('yelp.csv')
print('data frame head:\n', yelp_df.head())
print('\n data frame tail:\n',yelp_df.tail)
print('\ndescribing the data:\n', yelp_df.describe())
print('\nThe yelp info is below: \n',yelp_df.info())

print ('\nThe text of the first two reviews:\n', yelp_df['text'][0:2])
print('\nThe text of the last two reviews:\n', yelp_df['text'][9998:10000])

# VISUALIZING THE DATASET:
# Adding another feature called length which counts the number of characters with spaces in each of the text feature
yelp_df['length'] = yelp_df['text'].apply(len)
print (yelp_df)

# Plotting the length of the text feature on a histogram
yelp_df['length'].plot(bins=1000,kind='hist')
plt.show()

print('\n Description of the length feature:\n', yelp_df['length'].describe())


print('\nShowing the text that has the maximum number of characters in the text feature:\n', yelp_df[yelp_df['length'] == 4997]['text'].iloc[0])
print('\nShowing the text that has the minimum number of characters in the text feature:\n',
      yelp_df[yelp_df['length'] == 1]['text'].iloc[0])
print('\n Showing the text that has mean number of characters in the text feature: \n',
      yelp_df[yelp_df['length'] == 710]['text'].iloc[0])

# To check how many reviews have 1 star, 2 star, 3 star , 4 star and 5 star reviews
one_star = yelp_df[yelp_df['stars']==1]
print('\n',len(one_star), 'reviews have one star')

two_stars = yelp_df[yelp_df['stars']==2]
print('\n',len(two_stars), 'reviews have two stars')

three_stars = yelp_df[yelp_df['stars']==3]
print('\n',len(three_stars), 'reviews have three stars')


four_stars = yelp_df[yelp_df['stars']==4]
print('\n',len(four_stars), 'reviews have four stars')


five_stars = yelp_df[yelp_df['stars']==5]
print('\n',len(five_stars), 'reviews have five stars')

# Showing the respective stars on reviews on countplot
fig = plt.figure()
plt.subplot(1,2,1)
sns.countplot(y='stars', data= yelp_df)
plt.subplot(1,2,2)
sns.countplot(x='stars', data= yelp_df)
plt.show()

# Showing different review stars on a histogram
g = sns.FacetGrid(data = yelp_df, col= 'stars', col_wrap=3)
g.map(plt.hist, 'length', bins= 20, color='r')
plt.show()

# To concatenate the reviews with one star and the 5 star.
yelp_df_1_5 = pd.concat([one_star,five_stars])
print('\n Concatenated one and five stars reviews:\n',yelp_df_1_5)
print('\n The info of the concatenated one and five stars reviews :\n', yelp_df_1_5.info())

# To calculate the percentage of the one star review
print('The percentage of the 1-star review is',
      (len(one_star)/len(yelp_df_1_5)*100),'%')
# To calculate the percentage of the five star review
print('The percentage of the 5-star review is',(len(five_stars)/len(yelp_df_1_5)*100),'%')

# Visualizing the number of 1-star and the 5-star reviews on the concatenated 1 and 5 stars review on a countplot
sns.countplot(yelp_df_1_5['stars'],label ='count')
plt.show()

# CREATING TRAINING AND TESTING DATASET/DATA CLEANING

# Removing punctuation
print('\nPunctuations to remove from our dataset\n',string.punctuation)
Test = "Hello Love, do you know to get to No. 19 chemin de l'epitaphe, 25000 Besancon France!?"
Test_punc_removed = [char for char in Test if char not in string.punctuation] # return char means char in the first char i wrote on the list.
print(Test_punc_removed)

# Joining them text together to get the original test without punctuations.
Test_punc_removed_join = ''.join(Test_punc_removed)
print(Test_punc_removed_join)

# REMOVING THE STOPWORDS FROM OUR STRING
print (stopwords.words('english'))
print(Test_punc_removed_join.split()) # splitting the words in other to remove the stopwords
Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in
                                 stopwords.words('english')]

sentence = 'Mummy is going to make some soup, the children are helping her, each Child is doing a job, Eze is fetching water, Ada is peeling the onions and Chinwe is making the fire!!'
sentence_punc_removed = [char for char in sentence if char not in string.punctuation]
print(sentence_punc_removed)

sentence_punc_removed_join = ''.join(sentence_punc_removed)
print(sentence_punc_removed_join)

print(sentence_punc_removed_join.split())
sentence_punc_removed_join_clean = [word for word in sentence_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
print(sentence_punc_removed_join_clean)

# PERFORMING A COUNT VECTORIZER ON OUR MODEL
# Count vectorizer converts all the words in our data into a matrix that repeats or shows the frequency of occurrence of various words.
sample = ['This is your first task', 'This is your second task', 'This is the third task', 'The first task links the second while the second and the third tasks are not related', 'Goodluck!']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample)
print('\n Get the feature name of the vectorized sample:\n ', vectorizer.get_feature_names())

print('The vectorized sample in an array format:',X.toarray())

sample1 = ['Why would you drink your water the same time I am drinking mine', 'You are supposed to say Wow! very good, we are drinking at the same time', 'Water is good for health', 'Water is life']
Y = vectorizer.fit_transform(sample1)
print('\n Get the feature name of the vectorized sample1:\n',vectorizer.get_feature_names())
print('The vectorized sample in an array format:', Y.toarray())

# APPLYING THE COUNT VECTORIZER EXAMPLE ON MY MODEL
def message_cleaning(message):
      Test_punc_removed = [char for char in message if char not in string.punctuation]
      Test_punc_removed_join = ''.join(Test_punc_removed)
      Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
      return Test_punc_removed_join_clean

yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)
print('\nThe cleaned up review\n:',yelp_df_clean[0]) # The cleaned up review
print('\n The original review:\n', yelp_df_1_5['text'][0]) # The original review

print('\n The description of the 1 and 5 stars review:\n' ,yelp_df_1_5['length'].describe())

# Cleaning up the median of the yelp_1_5 review which is equal to 662
print('\n Showing the text of the median of the 1 and 5 star review: \n  ',yelp_df_1_5[yelp_df_1_5['length']==662]['text'].iloc[0])
print('\n Showing the index of the median of the 1 and 5 star review: \n', yelp_df_1_5[yelp_df_1_5['length']==662]['text'])

print(yelp_df_clean[3571])

# APPLYING COUNTVECTORIZER TO THE OUR MODEL
vectorizer = CountVectorizer(analyzer=message_cleaning)
yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])
print('\n feature names of the countvectorizer:\n',vectorizer.get_feature_names())
print('\n Array showing the Count Vectorized yelp dataset:\n',yelp_countvectorizer.toarray())

# TRAINING THE MODEL WITH ALL DATASET WITH NAIVE_BAYES MULTINOMIALNB
NB_classifier = MultinomialNB()
label = yelp_df_1_5['stars'].values
NB_classifier.fit(yelp_countvectorizer,label) # The fit method takes the yelp_countvectorizer as the input and the label as the output

# RANDOM EXAMPLES TO TEST AND ASCERTAIN THE PERFORMANCE OF THE MODEL
testing_sample = ['amazing food, highly recommended!!']
testing_sample1 = ['shit food, made me sick!!']

testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
print('\n The result of the random test sample: \n',test_predict)

testing_sample_countvectorizer = vectorizer.transform(testing_sample1)
test_predict1 = NB_classifier.predict(testing_sample_countvectorizer)
print('\n The result of the random test sample: \n',test_predict1)

# DIVIDING THE DATA INTO TRAINING AND TESTING PRIOR TO TESTING
X = yelp_countvectorizer
y = label
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
print('\n The shape of X(The yelp countvectorizer):\n', X.shape)
print('\n The shape of y(label):\n', y.shape)

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train,y_train)

# EVALUATING THE MODEL
# evaluating the model by comparing the Y_train and the Y_predict_train
y_predict_train = NB_classifier.predict(X_train)
cm = confusion_matrix(y_train,y_predict_train)
sns.heatmap(cm,annot=True)
plt.show()

# evaluating the model by comparing the Y_test and the Y_predict_test
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test,y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()

print('\nThe classification report of the model:\n', classification_report(y_test,y_predict_test))

# LETS ADD ADDITIONAL FEATURE TF-IDF
yelp_tfidf = TfidfTransformer().fit_transform(yelp_countvectorizer)
print('\nThe shape of the Yelp_tfidf:\n',yelp_tfidf.shape)
print(yelp_tfidf[:,:])
X = yelp_tfidf
y= label

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.1)
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train,y_train)

# EVALUATING THE MODEL
# evaluating the model by comparing the Y_train and the Y_predict_train
y_predict_train = NB_classifier.predict(X_train)
cm = confusion_matrix(y_train,y_predict_train)
sns.heatmap(cm,annot=True)
plt.show()

# evaluating the model by comparing the Y_test and the Y_predict_test
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test,y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()
































