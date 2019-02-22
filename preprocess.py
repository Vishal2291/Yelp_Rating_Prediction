# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:08:57 2019

@author: Vishal Boinapalli
"""
import nltk
import re  

nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer 
 
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer 
from nltk.text import Text
nltk.download('gutenberg')

dataset = pd.read_csv('C:/Users/Pooja Reddy/Documents/Major Project/yelp.csv',sep=",",usecols=(3,4),nrows=5)
output=[]
for i in range(0, 5):  
      
    # column : "Review", row ith 
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])  
      
    # convert all cases to lower cases 
    review = review.lower()  
      
    # split to array(default delimiter is " ") 
    review = review.split()  
      
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = PorterStemmer()  
      
    # loop for stemming each word 
    # in string array at ith row     
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]  
                  
    # append each string to create 
    # array of clean text  
    output.append(review)
    
print(dataset)
print(type(output))
for i in output:
    print(i)
print()

dt = Text(nltk.corpus.gutenberg.words('C:/Users/Pooja Reddy/Documents/Major Project/yelp.csv'))
print()

fdist = nltk.FreqDist(dt)
fdist.plot(50, cumulative=True)

