#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Denzel Tan
# #### Student ID: s3900098
# 
# Date: 19/9/2021
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# 

# ### The Dataset

# The Data we are working with is Job Advertisements.
# 
# There are approximately 55449 job ads we have to process.
# 
# Each job ad comes in the form of:
# 1. Title
# 2. WebIndex
# 3. Company
# 4. Description
# 
# Each job ad is also stored in its respective folder indicitive of the jobs category.
# 
# There are 8 categories:
# 1. Accounting_Finance
# 2. Engineering
# 3. Healthcare_Nursing
# 4. Hospitality_Catering
# 5. IT
# 6. PR_Advertising_Marketing
# 7. Sales
# 8. Teaching

# 

# 

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
import pandas as pd
import numpy as np
import re
from sklearn.datasets import load_files
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from nltk.probability import *
from nltk.util import ngrams


# ### 1. Examining and loading data
# 
# We will:
# - Examine the data folder, including the categories and job advertisment txt documents
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 
# There are 8 folders in total, each representing a category of job listings:
# - Accounting_Finance
# - Engineering
# - Healthcare_Nursing
# - Hospitality_Catering
# - IT
# - PR_Advertising_Marketing
# - Sales
# - Teaching
# 
# Inside the folder are text files, containing the:
# - Title
# - WebIndex
# - Company
# - Description

# #### 1.1 Looking at the Data

# In[2]:


# Code to inspect the provided data file...
job_data =load_files(r'data')


# In[3]:


# This is the structure of the dictionary that load_files gives us
# We have 5 sections
[print(i) for i in job_data]


# In[4]:


job_data['filenames']


# In[5]:


set(job_data['target'])


# In[6]:


# This means there are 8 folders, each number allocated to the respective category.
job_data['target_names']


# In[7]:


job_data['target'][:10]


# In[8]:


# Defining it to variables for easier calling
data, catg = job_data['data'], job_data['target']
num = 5
#print(data[num])
catg[num]


# In[9]:


# so the Lengths are the same at 55449, which is good.
print(len(data))
print(len(catg))


# In[10]:


data[0]


# #### 1.2 Extracting the desired information

# Lets extract the Description and WebIndex now

# In[11]:


# Converting the items in the list to a string
# because we need to extract parts with regex, and that requires a string.
data_str = [str(file) for file in data]
type(data_str[0])


# Now we need to create a list of just the descriptions.

# In[12]:


# Extract only the description
des_data = [re.findall(r'Description: (.*)',job) for job in data_str]
des_data[1]


# In[13]:


# Flatten the list of (single) lists to now a
# list of strings (the job description)
descr = []
for lis in des_data:
    for job_des in lis:
        descr.append(job_des)
print(len(descr))
type(descr[0])


# Yup looks correct. We now have a list of strings, each string being the job description.

# Done. Lets move on to tokenization.

# ### 2. Pre-processing data
# Perform the required text pre-processing steps.

# ...... Sections and code blocks on basic text pre-processing
# 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# #### 2.1 Tokenize each Job Description

# Creating our Tokenize Function:

# In[14]:


from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain


def tokenizeReview(review):
    """
        This function first convert all words to lowercases, 
        it then segment the raw review into sentences and tokenize each sentences 
        and convert the review to a list of tokens.
    """        
    #review = raw_review.decode('utf-8') # convert the bytes-like object to python string, need this before we apply any pattern search on it
    nl_review = review.lower() # cover all words to lowercase
    
    # segament into sentences
    sentences = sent_tokenize(nl_review)
    
    # tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    # merge them into a list of tokens
    tokenised_review = list(chain.from_iterable(token_lists))
    return tokenised_review


# Now lets create a a function which shows some general statistics:

# In[15]:


# Creating function for basic statistics
def stats_print(tk_reviews):
    words = list(chain.from_iterable(tk_reviews)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of reviews:", len(tk_reviews))
    lens = [len(article) for article in tk_reviews]
    print("Average review length:", np.mean(lens))
    print("Maximun review length:", np.max(lens))
    print("Minimun review length:", np.min(lens))
    print("Standard deviation of review length:", np.std(lens))


# In[16]:


# Checking what it looks like
descr[0]


# In[17]:


# TOKENIZING every job description
tk_descr = [tokenizeReview(des) for des in descr]


# In[18]:


stats_print(tk_descr)


# In[19]:


len(tk_descr)


# In[20]:


tk_descr[0]


# Nice. So now we have tokenized every job description

# #### 2.2 Remove words with <2 Length

# In[21]:


# Testing with the first one
for word in tk_descr[0]:
    if len(word)<2:
        print(word)


# In[22]:


# Removing the words with loop
tk_descr = [[word for word in des if len(word) >= 2] for des in tk_descr]


# In[23]:


# test again
for word in tk_descr[0]:
    if len(word)<2:
        print(word)


# Great it doesn't return anything. We got rid of them all.

# #### 2.3 Removing Stopwords

# In[24]:


# Extracting the stopwords from the .txt file and making sure the '\n' isnt included
with open('stopwords_en.txt','r') as f:
    lines = f.read()
    stopwords = lines.splitlines()
stopwords


# Ok stopwords extracted into list. Lets get to removing them now.

# In[25]:


# Finding the words that would be kept when we delete using the stopwords list
[word for word in tk_descr[0] if word not in stopwords]


# In[26]:


len(tk_descr[0])


# In[27]:


# Now REMOVING ALL stopwords
set_stopwords = set(stopwords)
tk_descr = [[word for word in des if word not in set_stopwords] for des in tk_descr]


# In[28]:


# Looks like the length decreased. The stopword removal worked.
len(tk_descr[0])


# In[29]:


stats_print(tk_descr)


# Also interesting note: The stopwords list actually has a duplicate in it, as found when the set was 1 less than the normal list. The duplicate is "would". However, for stopword removal, a list or set shouldn't impact the final product, so a list was used to be safe. 
# 
# Note that if the list was much bigger, a set might have needed to be used for speed

# In[30]:


kid = FreqDist(stopwords)


# In[31]:


kid.most_common()


# #### 2.4 Remove words that appear once in the document collection

# In[32]:


words = list(chain.from_iterable(tk_descr)) # we put all the tokens in the corpus in a single list
vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words


# In[33]:


term_fd = FreqDist(words)


# In[34]:


# Checking the last 3 on the most common list
term_fd.most_common()[-3:]


# In[35]:


# Appending all words that occur once to a list.
occur_once = []
occur = 1
for term, occurances in term_fd.items(): # Since FreqDist gives a dictionary, we need to search each key:value and 
    if occurances == occur: # find values that match to what we want.
        occur_once.append(term)


# In[36]:


# Validated. Looks like our list works. 
occur_once[-3:]


# In[37]:


len(occur_once)


# In[38]:


occur_once_set = set(occur_once)


# In[39]:


len(occur_once_set)


# Now we have a list of words (single occurance) that we want to remove. Lets follow the same process as the stopwords.

# In[40]:


# Now REMOVING ALL words that occur once
tk_descr = [[word for word in des if word not in occur_once_set] for des in tk_descr]


# In[41]:


len(tk_descr)


# In[42]:


stats_print(tk_descr)


# In[43]:


89431-49237


# Nice. We got rid of the single occurance words, validated by our simple math.
# 
# Note: A set needed to be used (not that it changed anything about the occur once list) but it was significently faster.

# #### 2.5 Remove the top 50 most frequent words based on Document Frequency

# In[44]:


# Grabbing the top 50 most common words based on document freq
words_2 = list(chain.from_iterable([set(des) for des in tk_descr]))
doc_fd = FreqDist(words_2)  # compute document frequency for each unique word/type
top_50 = doc_fd.most_common(50)
top_50


# In[45]:


# Testing to see if the term frequency would be different
test = list(chain.from_iterable([des for des in tk_descr]))
doc_fd2 = FreqDist(test)
doc_fd2.most_common(5)


# Yup we can clearly see that there is a huge difference using the document frequency instead. We have now validated it is working.

# In[46]:


# Test to see if list is correct
top_50[0][0]


# In[47]:


# Length looks correct
len(top_50)


# In[48]:


# Making a list of the top 50 words itself
top_50_list = []
for i in range(0,len(top_50)):
    top_50_list.append(top_50[i][0])
top_50_list


# In[49]:


# Initial length check
len(tk_descr[0])


# In[50]:


# Removing the top 50 words now
top_50_list_set = set(top_50_list)
tk_descr = [[word for word in des if word not in top_50_list_set] for des in tk_descr]


# In[51]:


# Validated. Looks like words got removed.
len(tk_descr[0])


# In[52]:


stats_print(tk_descr)


# Hooray. We removed the top 50 most frequent words from every description.

# #### 2.6 Extract the Top 10 Bigrams based on term frequency (save to .txt file)

# In[53]:


# Updating the words and vocab to latest.
words = list(chain.from_iterable(tk_descr)) # we put all the tokens in the corpus in a single list
vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words


# In[54]:


# Setting up our bigrams
bigrams = ngrams(words, n =2)
fdbigram = FreqDist(bigrams)


# In[55]:


# Finding top 10 most common Bigrams
top_10 = fdbigram.most_common(10)
top_10


# In[56]:


# Making list of bigrams, combined into string with its count.
top_10_combined = []
for i in range(0,len(top_10)):
    top_10_combined.append( " ".join(top_10[i][0]) + ',' + str(top_10[i][1]))


# In[57]:


top_10_combined


# Ok we got it into a list. Now we just have to save it later.

# In[58]:


stats_print(tk_descr)


# #### 2.7 Build Vocab list and cleaned Job Ad list

# In[59]:


# Updating the words and vocab to latest.
words = list(chain.from_iterable(tk_descr)) # we put all the tokens in the corpus in a single list
vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
vocab_list = list(vocab)


# In[60]:


len(vocab_list)


# In[61]:


# Sorting our list into alphabetical order
vocab_list = sorted(vocab_list)
vocab_list


# In[62]:


# Testing if correct
vocab_list[0]


# In[63]:


# Saving the vocab and its Index to a list in the desired format.
full_vocab = []
for i in range(0,len(vocab_list)):
    full_vocab.append(vocab_list[i] +":"+ str(i))


# In[64]:


# Testing to compare below if our list worked and appending the correct form.
i = 21
vocab_list[i] +":"+ str(i)


# In[65]:


full_vocab


# ## Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt
# - bigram.txt
# - job_ads.txt

# #### Saving Bigrams

# In[66]:


# Saving Bigrams Function
def save_bigrams(bigramFilename,bigrams):
    out_file = open(bigramFilename, 'w') # creates a txt file and open to save Bigrams
    string = "\n".join([str(s) for s in bigrams])
    out_file.write(string)
    out_file.close() # close the file 


# In[67]:


# Saving Bigrams
save_bigrams('bigram.txt',top_10_combined)


# #### Saving Vocab
# 
# Lets just use the same function as the Bigrams, since it does exactly the same thing.
# 
# And to save time and be effecient! No point reinventing the wheel right?
# 
# 

# In[68]:


# Saving Vocab
save_bigrams('vocab.txt',full_vocab)


# ### Saving job advertisements

# We have 5 sections required for our output. Lets put them each into a list first.

# #### 1. Id

# In[69]:


job_data['filenames'][0]


# In[70]:


# Extracting the 5 digit ID from each filename
id_list = []
for i in range(0,len(job_data['filenames'])):
    id_list.append(re.findall(r'_(\d{5}).txt',job_data['filenames'][i]))
len(id_list)


# In[71]:


# Flatten the list of lists
id_list_flat = list(chain.from_iterable(id_list))
id_list_flat


# #### 2. Category

# In[72]:


# Use regex to pull out the category from filename
cat_list = []
for i in range(0,len(job_data['filenames'])):
    cat_list.append(re.findall(r'/(\w+)/',job_data['filenames'][i]))
len(cat_list)


# In[73]:


# Flatten list of lists into just a list, then find the set to confirm if all Categories were picked up.
cat_list_flat = list(chain.from_iterable(cat_list))
set(cat_list_flat)


# Looks like we got all categories captured. That verifies it for us now.

# #### 3. WebIndex

# In[74]:


# Making list of web index's 
web_list = []
for i in range(0,len(job_data["data"])):
    web_list.append(re.findall(r'Webindex: (\d{8})',str(job_data["data"][i])))
len(web_list)


# In[75]:


# Flatten the list of lists
web_list_flat = list(chain.from_iterable(web_list))
web_list_flat


# #### 4. Title

# In[76]:


# The Regex Pattern for extracting the Title
re.findall(r'Title: (.+)\\nWebindex:',str(job_data["data"][21]))


# In[77]:


# Making list of Titles
title_list = []
for i in range(0,len(job_data["data"])):
    title_list.append(re.findall(r'Title: (.+)\\nWebindex:',str(job_data["data"][i])))
len(title_list)


# In[78]:


# Flatten the list of lists
title_list_flat = list(chain.from_iterable(title_list))
title_list_flat


# #### 5. Description 

# In[79]:


# Joining all description tokens togather into a single string
des_string = [' '.join(token) for token in tk_descr]
len(des_string)


# In[80]:


# Now a list of JUST the description
des_string[0:2]


# In[81]:


# Creating a HUGE list of every single description + its corresponding info. 
all_info = []
for i in range(0,len(job_data["data"])):
    
    string = "ID: " + id_list_flat[i] + "\nCategory: " + cat_list_flat[i] + "\nWebindex: " + web_list_flat[i] + "\nTitle: " + title_list_flat[i] + "\nDescription: " + des_string[i]
    all_info.append(string)
len(all_info)


# In[82]:


# Checking our huge list complete with all info
all_info[:2]


# ### Now we save the data.

# In[83]:


def save_all(filename,contents):
    out_file = open(filename, 'w') # creates a txt file and open to save the ads
    out_file.write("\n".join([info for info in contents])) # Write the content
    out_file.close() # close the file


# In[84]:


save_all("job_ads.txt",all_info)


# ## Summary

# Overall, we exported 3 files.
# 
# The main difficulty was extracting the description out of the original data file, however that was solved via the very useful Regex.
# 
# I learnt that Sets can be very useful, and can be significently faster than lists in some applications.

# ## Couple of notes for all code blocks in this notebook
# - please provide proper comment on your code
# - Please re-start and run all cells to make sure codes are runable and include your output in the submission.   
# <span style="color: red"> This markdown block can be removed once the task is completed. </span>

# In[85]:


stats_print(tk_descr)


# In[ ]:




