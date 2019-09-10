#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# In[2]:


#changing directory
os.chdir('C:\\Users\\Acer\\OneDrive\\Desktop\\Language_1')


# In[3]:


#importing dataset one
dataset1 = pd.read_csv('Round1_Problem1-of-2_Dataset_amazon_cells_labelled.tsv', delimiter = '\t', quoting = 3,header=-1,na_values=[".com"])


# In[4]:


#importing dataset two
dataset2 = pd.read_csv('Round1_Problem1-of-2_Dataset_imdb_labelled.tsv', delimiter = '\t', quoting = 3,header=-1,na_values=[".com"])


# In[5]:


# renaming of columns
dataset1.rename(columns={0: "review1", 1: "label1"},inplace=True)


# In[6]:


#renaming of columns
dataset2.rename(columns={0: "review2", 1: "label2"},inplace=True)


# In[7]:


import re
import nltk
from nltk.stem import WordNetLemmatizer
import emoji
from nltk.corpus import stopwords
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z!.,]', ' ', dataset1['review1'][i]) #tokenization(removing every punctuation mark and adding space)
    review = review.lower()#converting capital to small letters
    review = review.split()#splitting the string to individual words
    review = [emoji.demojize(word) for word in review]#convert emoticons to text
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]#removing useless words like the , then which are not required and also lemmatizing the remaining words like loved to love etc 
    review = ' '.join(review) #then again joining the remaining words with space between them
    corpus.append(review)# adding the objects to corpus


# In[8]:


import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
corpus1 = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z!.,]', ' ', dataset2['review2'][i]) #tokenization(removing every punctuation mark and adding space)
    review = review.lower()#converting capital to small letters
    review = review.split()#splitting the string to individual words
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]#removing useless words like the , then which are not required and also lemmatizing the remaining words like loved to love etc 
    review = ' '.join(review) #then again joining the remaining words with space between them
    corpus1.append(review)# adding the objects to corpus


# In[9]:


corpus


# In[10]:


corpus1


# In[11]:


#convert the data into vector forms using Word2Vec with vector size of 20
from gensim.models import Word2Vec
model1 = Word2Vec(corpus, min_count=1,size=20)


# In[12]:


model1


# In[13]:


#convert the data into vector forms using Word2Vec with vector size of 20
from gensim.models import Word2Vec
model2 = Word2Vec(corpus1, min_count=1,size=20)


# In[14]:


model2


# In[15]:


#converting list of output to dataframe
output1=pd.DataFrame(corpus)
output2=pd.DataFrame(corpus1)


# In[16]:


output1.rename(columns={0: "output_review1"},inplace=True)


# In[17]:


output2.rename(columns={0: "output_review2"},inplace=True)


# In[18]:


output=pd.concat([output1,dataset1,output2,dataset2],axis=1)


# In[19]:


output=output.drop(columns=['review1','review2'])


# In[20]:


output


# In[21]:


#creating csv file
output.to_csv(r'C:\Users\Acer\OneDrive\Desktop\Language_1\output.csv',index=False)


# In[ ]:




