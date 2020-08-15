#!/usr/bin/env python
# coding: utf-8

# # Simply use (Shift+Enter) to execute each of these code blocks in sequence, to generate vectors for the corpora used.

# In[ ]:


#import the following libraries. Install them, if not present on your system.
import fasttext
import os
import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


# In[ ]:


#Used to remove html markups, etc

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


# In[ ]:


#Function that generates a list of elements, where each element here is the full path to each of the text files.

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


# In[ ]:


#Functions used for preprocessing textual data

def replace_contractions(text):
    """Replaces contractions (it's -> it is)"""
    return contractions.fix(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words


# In[ ]:


#Generating the combined text file


#Creating a list of the filenames required
#all_files = getListOfFiles('/path/to/Corpora')

all_files = getListOfFiles('/home/nishanthsanjeev/Harvard/Corpora')


# Here, the 'Corpora' file contains the text files that will be opened and combined into the final text file.

# Kindly make sure that the only files in this directory are the text files that are being considered, although
# they can be stored (separately or not) in any sub-directory, as long as they are all within the 'Corpora' directory. 


# In[ ]:


#Generating the new, combined text file.


with open('\Harvard\file_combined.txt', 'w') as outfile:
    for names in all_files:
        print(names)
        with open(names,'r') as infile:
            outfile.write(infile.read())
            
        outfile.write("\n")


# In[ ]:


#the 'combined_data' variable now stores the contents of 'file_combined.txt'

combined_data = open('\Harvard\file_combined.txt','r').read()


# In[ ]:


comb_cl = re.sub(r"\\[']\d*","'",combined_data)


#A common occurence of garbage occurring in the txt files, that I was able to remove:

# "I\'92ve got a sweet tooth!"

# This regex converts this sentence to:

# "I've got a sweet tooth!"


# In[ ]:


#Calling the denoise() function on the data
comb_cl_den = denoise_text(comb_cl)


# In[ ]:


#Tokenizing into sentences -> This converts the huge paragraph of text into a list of sentences.

sentences = nltk.sent_tokenize(comb_cl_den)


# In[ ]:


#Replacing contractions, i.e. it's -> it is, et cetera

sentences_mod = []
for x in sentences:
    x = replace_contractions(x)
    sentences_mod.append(x)
    


# In[ ]:


#Tokenizing each sentence into it's constituent words.

# For example,

# 'I like ice cream'

# becomes a list, with each of its elements being words.

# -> ['I','like','ice','cream']



words = []
for el in sentences_mod:
    words.append(nltk.word_tokenize(el))


# Checking if everything went okay:
# 
# Assume, your input text was:
# 
# "I like ice cream. Let's go buy some!"
# 
# At this step, the 'words' variable must hold a list of lists.
# Each of these lists hold tokenized sentences.
# 
# Therefore, we should get:
# 
# [['I', 'like', 'ice', 'cream', '.'], ['let', 'us', 'go', 'buy', 'some', '!']]
# 
# It is important that the data is in the list of lists format, as this is the required input format for Word2Vec.

# In[ ]:


#Removing non-ascii words
non_asc = []
for s in words:
    temp = remove_non_ascii(s)
    non_asc.append(temp)


# In[ ]:


#Converting to lowercase
lower_c = []
for s in non_asc:
    temp = to_lowercase(s)
    lower_c.append(temp)


# In[ ]:


#Removing punctutation
punct_g = []
for s in lower_c:
    temp = remove_punctuation(s)
    punct_g.append(temp)


# In[ ]:



#     Final pre-processing step: Lemmatizing the data. (Converting tokens to their root forms)
    
#     E.g.: Converts 'running','ran',etc to 'run'


train_data = []
for f in punct_g:
    f = lemmatize_verbs(f)
    train_data.append(f)#stop words remain


# In[ ]:


embedding_size = 60
window_size = 40
min_word = 5
down_sampling = 1e-2


# In[ ]:


from gensim.models.fasttext import FastText

ft_model = FastText(train_data,
                      size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                    sg=1,
                      iter=10)


# In[ ]:


semantically_similar_words = {words: [item[0] for item in ft_model.wv.most_similar([words], topn=5)]
                  for words in ['kitchen', 'death', 'king', 'queen', 'strong', 'weak','woman','man']}

for k,v in semantically_similar_words.items():
    print(k+":"+str(v))


# In[ ]:


ft_model.similarity("annabeth","percy")


# In[ ]:


ft_model.wv.save_word2vec_format('FTvectors')

