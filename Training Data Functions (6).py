#!/usr/bin/env python
# coding: utf-8
Training Data for Pattern Recognition Group Theory work 
# # 0 Some Preliminary Functions

# In[4]:


import numpy as np
import numpy.random as npr
import tensorflow.keras as keras
import matplotlib.pyplot as plt 
import random
import unittest  
import time   

from sklearn.linear_model import LinearRegression
from scipy.stats import gamma
from collections import Counter
from scipy.stats import mode
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split 


# In[5]:


def reduce_and_normalise(word): # Tested in another document
    """
    Reduces and normalizes a word represented as a list of tuples.
    Returns the reduced and normalized word.
    """
    reduced = []
    
    for generator, power in word:
        if reduced and reduced[-1][0] == generator:
            combined_power = reduced[-1][1] + power
            
            if combined_power != 0:
                reduced[-1] = (generator, combined_power)
            else:
                reduced.pop()
        else:
            reduced.append((generator, power))
    
    normalised = []
    
    for gen, power in reduced:
        if power < 0:
            # If power is negative, convert it to -1 by repeating the generator
            normalised.extend([(gen, -1)] * abs(power))
        elif power > 0:
            # If power is positive, convert it to 1 by repeating the generator
            normalised.extend([(gen, 1)] * power)
        else:
            # If power is zero, ignore it
            pass
    
    return normalised  

def red_word_length(word):
    
    # Reduce the word to condensed tuple form 
    reduced =  reduce_and_normalise(word)
    
    # Calculate the sum of absolute values of the second part of each tuple
    result = sum(abs(w[1]) for w in reduced) 
    return result


# # 1.1 No Return Walk

# In[6]:


def inverse_letter(letter):  
    return((letter[0],-letter[1])) 

# function that returns a cyclically reduced word 

def cyc_reduce(word1):
    word=reduce_and_normalise(word1)
    while len(word) >= 2 and word[0] == inverse_letter(word[-1]):
        # Remove the first and last letters
        word = word[1:-1]

    return word 

def cyc_reduce_length(word): 
    return len(cyc_reduce(word))

def no_return_walk(l):
    
    def inverse_letter(letter):  
        return((letter[0],-letter[1]))
    
    result = []

    while len(result) < l:
        if len(result) == 0:
            # With probability 1/4, append one of 
            choice = random.choices([("a",1), ("a",-1), ("b",1), ("b",-1)], weights=[1/4] * 4)[0]
            result.append(choice)
        else:
            last_choice = inverse_letter(result[-1])
            # Determine the valid choices based on the last choice
            valid_choices = [i for i in [("a",1), ("a",-1), ("b",1), ("b",-1)] if i != last_choice]
            # With probability 1/3, append one of the valid choices
            choice = random.choice(valid_choices)
            result.append(choice)

    return result


# # 1.1.1 Unit Test

# In[7]:


y=no_return_walk(1000) # test exploring randomness 
count_a=0
count_b=0 
count_a_inv=0 
count_b_inv=0
for letter in y: 
    if letter == ("a",1): 
        count_a+=1 
    elif letter ==("a",-1): 
        count_a_inv+=1  
    elif letter ==("b",1): 
        count_b+=1
    elif letter ==("b",-1): 
        count_b_inv+=1
        
print(count_a, count_b, count_a_inv, count_b_inv)


class TestNotebook(unittest.TestCase):
    def test_no_return(self):
        result = no_return_walk(1000)
        x = True  

        for i in range(1, len(result)):  
            if result[i] == inverse_letter(result[i - 1]):
                x = False
                break 

        self.assertTrue(x)

unittest.main(argv=[''], verbosity=2, exit=False)


# # 1.2 Cyclically Reduced No Return Walk

# In[8]:


def no_return_cyclic_reduced(l):
    if l == 0:
        return []
    
    if l == 1:
        generators = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]
        choice = random.choice(generators)
        return [choice]
    
    result = no_return_walk(l - 1)
    generators = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]
    first_choice_inverse = inverse_letter(result[0])
    last_choice_inverse = inverse_letter(result[-1])
    
    generators.remove(first_choice_inverse)
    
    if first_choice_inverse != last_choice_inverse:
        generators.remove(last_choice_inverse)
    else:
        pass

    n = len(generators)
    choice = random.choices(generators, weights=[1/n] * n)[0]
    result.append(choice)
    
    return result

for i in range(1,10):
    print(no_return_cyclic_reduced(i))


# # 1.2.1 Unit Test

# In[9]:



class TestNotebook(unittest.TestCase):
    
    def test_non_cyclic(self): 
        y=no_return_cyclic_reduced(100) 
        self.assertNotEqual(y[0],y[-1])
        
unittest.main(argv=[''], verbosity=2, exit=False)


# # 1.3 List of Cyclically Reduced Words

# In[10]:


def list_of_cyclic_reduced_words(n, l):
    result = []

    for _ in range(n):
        word = no_return_cyclic_reduced(l)
        result.append(word)
        
    return result


# In[11]:


y=list_of_cyclic_reduced_words(20,3) 
print(y) 

for x in y: 
    if x[0]==inverse_letter(x[-1]):
        print("we found a bad one", x)


# In[ ]:





# In[ ]:





# # 1.3.1 Unit Test 

# In[12]:



class TestNotebook(unittest.TestCase):
    
    def test_non_cyclic_in_list(self): 
        list_words=list_of_cyclic_reduced_words(1000,10)
        for words in list_words:
            self.assertNotEqual(words[0],inverse_letter(words[-1]))
        
unittest.main(argv=[''], verbosity=2, exit=False)


# # 2 Whitehead's Algorithm

# I think for F_2 we will have 19 functions to search through - that is the 7 permutations functions (we don't count the identity), then 12 Nielsen type functions (4 of which are of the conjugation type).

# In[13]:


# Type 1 Permutations 
# No need to code since Whitehead type 1 aut don't change the length of the word. 
# Left for posterity. 
# Can be used for projection onto fundamental domain!

def perm_1_letter(word): 
    if word == ("a", 1): 
        return ("b", 1) 
    elif word == ("a", -1):  
        return ("b", -1) 
    elif word == ("b", 1):  
        return ("a", 1) 
    elif word == ("b", -1):  
        return ("a", -1)    


def perm_1(word): 
    result = [] 
    for letters in word: 
        result.append(perm_1_letter(letters)) 
    return result 

def perm_2_letter(word): 
    if word == ("a", 1): 
        return ("b", 1) 
    elif word == ("a", -1):  
        return ("b", -1) 
    elif word == ("b", 1):  
        return ("a", -1) 
    elif word == ("b", -1):  
        return ("a", 1)

def perm_2(word): 
    result = [] 
    for letters in word: 
        result.append(perm_2_letter(letters)) 
    return result 

def perm_3_letter(word): 
    if word == ("a", 1): 
        return ("b", -1) 
    elif word == ("a", -1):  
        return ("b", 1) 
    elif word == ("b", 1):  
        return ("a", 1) 
    elif word == ("b", -1):  
        return ("a", -1)

def perm_3(word): 
    result = [] 
    for letters in word: 
        result.append(perm_3_letter(letters)) 
    return result 

def perm_4_letter(word): 
    if word == ("a", 1): 
        return ("b", -1) 
    elif word == ("a", -1):  
        return ("b", 1) 
    elif word == ("b", 1):  
        return ("a", -1) 
    elif word == ("b", -1):  
        return ("a", 1)

def perm_4(word): 
    result = [] 
    for letters in word: 
        result.append(perm_4_letter(letters)) 
    return result 

def perm_5_letter(word): 
    if word == ("a", 1): 
        return ("a", -1) 
    elif word == ("a", -1):  
        return ("a", 1) 
    elif word == ("b", 1):  
        return ("b", 1) 
    elif word == ("b", -1):  
        return ("b", -1)

def perm_5(word): 
    result = [] 
    for letters in word: 
        result.append(perm_5_letter(letters)) 
    return result 

def perm_6_letter(word): 
    if word == ("a", 1): 
        return ("a", -1) 
    elif word == ("a", -1):  
        return ("a", 1) 
    elif word == ("b", 1):  
        return ("b", -1) 
    elif word == ("b", -1):  
        return ("b", 1)

def perm_6(word): 
    result = [] 
    for letters in word: 
        result.append(perm_6_letter(letters)) 
    return result 

def perm_7_letter(word): 
    if word == ("a", 1): 
        return ("a", 1) 
    elif word == ("a", -1):  
        return ("a", -1) 
    elif word == ("b", 1):  
        return ("b", -1) 
    elif word == ("b", -1):  
        return ("b", 1)

def perm_7(word): 
    result = [] 
    for letters in word: 
        result.append(perm_7_letter(letters)) 
    return result


# In[14]:


Type_1_functions=[perm_1,perm_2,perm_3,perm_4,perm_5,perm_6,perm_7]


# In[15]:


def nielsen_1_letter(word): #This needs checking through
        if word == ("a", 1):
            return [("b", -1), ("a", 1)]
        elif word == ("a", -1):
            return [("a", -1), ("b", 1)]
        elif word == ("b", 1):
            return [("b", 1)]
        elif word == ("b", -1):
            return [("b", -1)]
    
def nielsen_1(word):
        result = []
        for letters in word:
            new_words = nielsen_1_letter(letters)
            for new_word in new_words:
                result.append(new_word)
        return result     

def nielsen_2_letter(word):
        if word == ("a", 1):
            return [("b", 1), ("a", 1)]
        elif word == ("a", -1):
            return [("a", -1), ("b", -1)]
        elif word == ("b", 1):
            return [("b", 1)]
        elif word == ("b", -1):
            return [("b", -1)]   

def nielsen_2(word):
        result = []
        for letters in word:
            new_words = nielsen_2_letter(letters)
            for new_word in new_words:
                result.append(new_word)
        return result 
    
def nielsen_3_letter(word):
        if word == ("a", 1):
            return [("a", 1), ("b", 1)]
        elif word == ("a", -1):
            return [("b", -1), ("a", -1)]
        elif word == ("b", 1):
            return [("b", 1)]
        elif word == ("b", -1):
            return [("b", -1)]    

def nielsen_3(word):
        result = []
        for letters in word:
            new_words = nielsen_3_letter(letters)
            for new_word in new_words:
                result.append(new_word)
        return result     

def nielsen_4_letter(word):
        if word == ("a", 1):
            return [("a", 1), ("b", -1)]
        elif word == ("a", -1):
            return [("b", 1), ("a", -1)]
        elif word == ("b", 1):
            return [("b", 1)]
        elif word == ("b", -1):
            return [("b", -1)]    

def nielsen_4(word):
        result = []
        for letters in word:
            new_words = nielsen_4_letter(letters)
            for new_word in new_words:
                result.append(new_word)
        return result     

def nielsen_5_letter(word):
        if word == ("a", 1):
            return [("a", 1)]
        elif word == ("a", -1):
            return [("a", -1)]
        elif word == ("b", 1):
            return [("a",1),("b", 1)]
        elif word == ("b", -1):
            return [("b", -1),("a",-1)]   

def nielsen_5(word):
        result = []
        for letters in word:
            new_words = nielsen_5_letter(letters)
            for new_word in new_words:
                result.append(new_word)
        return result      

def nielsen_6_letter(word):
        if word == ("a", 1):
            return [("a", 1)]
        elif word == ("a", -1):
            return [("a", -1)]
        elif word == ("b", 1):
            return [("a",-1),("b", 1)]
        elif word == ("b", -1):
            return [("b", -1),("a",1)]
    
def nielsen_6(word):
        result = []
        for letters in word:
            new_words = nielsen_6_letter(letters)
            for new_word in new_words:
                result.append(new_word)
        return result      

def nielsen_7_letter(word):
        if word == ("a", 1):
            return [("a", 1)]
        elif word == ("a", -1):
            return [("a", -1)]
        elif word == ("b", 1):
            return [("b",1),("a", 1)]
        elif word == ("b", -1):
            return [("a", -1),("b",-1)]
    
    
def nielsen_7(word):
        result = []
        for letters in word:
            new_words = nielsen_7_letter(letters)
            for new_word in new_words:
                result.append(new_word)
        return result  
    
def nielsen_8_letter(word):
        if word == ("a", 1):
            return [("a", 1)]
        elif word == ("a", -1):
            return [("a", -1)]
        elif word == ("b", 1):
            return [("b",-1),("a", 1)]
        elif word == ("b", -1):
            return [("a", -1),("b",1)]   

def nielsen_8(word):
        result = []
        for letters in word:
            new_words = nielsen_8_letter(letters)
            for new_word in new_words:
                result.append(new_word)
        return result 
        


# In[16]:


nielsen_7([("a",1),("b",1),("a",-1),("b",-1)])


# In[17]:


Nielsen_transforms=[nielsen_1,nielsen_2,nielsen_3,nielsen_4,nielsen_5,nielsen_6,nielsen_7,nielsen_8]


# In[18]:


# conjugation functions. 


# In[19]:


def conj_1_letter(word):
    if word == ("a", 1):
        return [("a", 1)]
    elif word == ("a", -1):
        return [("a", -1)]
    elif word == ("b", 1):
        return [("a", -1), ("b", 1), ("a", 1)]
    elif word == ("b", -1):
        return [("a", -1), ("b", -1), ("a", 1)]

def conj_1(word):
    result = []
    for letters in word:
        new_words = conj_1_letter(letters)
        for new_word in new_words:
            result.append(new_word)
    return result  

def conj_2_letter(word):
    if word == ("a", 1):
        return [("a", 1)]
    elif word == ("a", -1):
        return [("a", -1)]
    elif word == ("b", 1):
        return [("a", 1), ("b", 1), ("a", -1)]
    elif word == ("b", -1):
        return [("a", 1), ("b", -1), ("a", -1)]

def conj_2(word):
    result = []
    for letters in word:
        new_words = conj_2_letter(letters)
        for new_word in new_words:
            result.append(new_word)
    return result

def conj_3_letter(word):
    if word == ("a", 1):
        return [("b", -1), ("a", 1), ("b", 1)]
    elif word == ("a", -1):
        return [("b", -1), ("a", -1), ("b", 1)]
    elif word == ("b", 1):
        return [("b", 1)]
    elif word == ("b", -1):
        return [("b", -1)]

def conj_3(word):
    result = []
    for letters in word:
        new_words = conj_3_letter(letters)
        for new_word in new_words:
            result.append(new_word)
    return result 

def conj_4_letter(word):
    if word == ("a", 1):
        return [("b", 1), ("a", 1), ("b", -1)]
    elif word == ("a", -1):
        return [("b", 1), ("a", -1), ("b", -1)]
    elif word == ("b", 1):
        return [("b", 1)]
    elif word == ("b", -1):
        return [("b", -1)]

def conj_4(word):
    result = []
    for letters in word:
        new_words = conj_4_letter(letters)
        for new_word in new_words:
            result.append(new_word)
    return result


# In[20]:


Conjugation_functions=[conj_1,conj_2,conj_3,conj_4] 


# In[21]:


All_functions = Conjugation_functions+Nielsen_transforms # Deleted type 1 transforms. 


# In[22]:


def words_getting_smaller(word):
    original_length = red_word_length(word)

    for func in All_functions:
        transformed_word = cyc_reduce(func(word))
        transformed_length = cyc_reduce_length(transformed_word)

        if transformed_length < original_length:
            return reduce_and_normalise(transformed_word)

    # If no function reduces the word length, return the original word
    return word #These functions need to be tested


# In[23]:


words_getting_smaller([("a",1),("b",1),("a",1)])


# In[ ]:





# In[24]:


class TestNotebook(unittest.TestCase): # Tests automorphisms reduce length of the word. 
    def test_length_red(self): 
        list_words=list_of_cyclic_reduced_words(1000,10) 
        for words in list_words: 
            y=True 
            if red_word_length(words)<red_word_length(words_getting_smaller(words)): 
                y=False 
            else:
                pass

            self.assertTrue(x)

unittest.main(argv=[''], verbosity=2, exit=False)


# In[25]:


def words_getting_bigger(word):
    original_length = red_word_length(word)

    # Shuffle the list of functions
    shuffled_functions = All_functions.copy()
    random.shuffle(shuffled_functions)

    for func in shuffled_functions:
        transformed_word = cyc_reduce(func(word))
        transformed_length = cyc_reduce_length(transformed_word)

        if transformed_length > original_length:
            return reduce_and_normalise(transformed_word)

    # If no function reduces the word length, return the original word
    return word


# In[26]:


words_getting_smaller([("a",1),("b",1),("a",1)])


# In[27]:


class TestNotebook(unittest.TestCase): # Tests automorphisms reduce length of the word. 
    def test_length_red(self): 
        list_words=list_of_cyclic_reduced_words(1000,10) 
        for words in list_words: 
            y=True 
            if red_word_length(words)>red_word_length(words_getting_bigger(words)): 
                y=False 
            else:
                pass

            self.assertTrue(x)

unittest.main(argv=[''], verbosity=2, exit=False)


# In[28]:


words_getting_bigger([("a",1)])


# In[29]:


def find_minimal_rep(word):
    original_length = red_word_length(word)
    
    for i in range(original_length):
        transformed_word = words_getting_smaller(word)
        transformed_length = red_word_length(transformed_word)
        
        if transformed_length == original_length:
            return transformed_word
        
        word = transformed_word
    
    return word


# In[30]:


find_minimal_rep([("b",1),("a",1)])


# # Short Training Data: exploring how often cyclically reduced words are minimal. 

# In[31]:


non_minimal_training_data = [] # generating non-minimal random cyclically reduced words

for length in range(1, 1001):
    words = list_of_cyclic_reduced_words(10, length)
    non_minimal_training_data.extend(words) 
    
print(len(non_minimal_training_data))


# In[32]:


minimal_training_data=[] # using whitehead algorithm to make minimum. 

for words in non_minimal_training_data: 
    minimal_words=find_minimal_rep(words)  
    minimal_training_data.append(minimal_words)

len(minimal_training_data)


# In[33]:


# checking the differences of the list. # Probably should iterate a few times?...

minimal_training_data 
non_minimal_training_data 

non_minimal_words=[]
for n in range(0,len(minimal_training_data)): 
    if minimal_training_data[n]!=non_minimal_training_data[n]: 
        non_minimal_words.append(non_minimal_training_data[n]) 
len(non_minimal_words)    


# In[34]:


list_of_frequencies=[] # length of non-minimal cyclically reduced words. 
for words in non_minimal_words: 
    list_of_frequencies.append(len(words)) 


# In[35]:


list_of_frequencies_array = np.array(list_of_frequencies) # some data analysis on the frequencies. 

# Calculate mean, mode, median, and standard deviation
mean_value = np.mean(list_of_frequencies_array)
mode_value = mode(list_of_frequencies_array)[0][0]  # Scipy's mode returns a ModeResult object
median_value = np.median(list_of_frequencies_array)
std_deviation = np.std(list_of_frequencies_array)

# Print the results
print(f"Mean: {mean_value}")
print(f"Mode: {mode_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_deviation}")


# In[36]:




# Use Counter to count the frequency of each number
counter = Counter(list_of_frequencies)

# Extract values and frequencies from Counter
values = list(counter.keys())
frequencies = list(counter.values())

# Create a bar chart
plt.bar(values, frequencies, label='Data')

# Fit a gamma distribution
a, loc, scale = gamma.fit(list_of_frequencies)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = gamma.pdf(x, a, loc, scale)
plt.plot(x, p * len(list_of_frequencies), 'r-', label='Gamma Fit')

plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Length of Non-Minimal Cyclically Reduced Words')
plt.legend()
plt.show


# # Producing Training Data

# First we need to produce 10,000 words: 10 for each length going from 1 to 1000. 

# In[37]:


non_minimal_training_data = []

for length in range(1, 1001):
    words = list_of_cyclic_reduced_words(10, length)
    non_minimal_training_data.extend(words) 
    
print(len(non_minimal_training_data))


# In[38]:


minimal_training_data=[] 

for words in non_minimal_training_data: 
    minimal_words=find_minimal_rep(words)  
    minimal_training_data.append(minimal_words)


# In[39]:


len(minimal_training_data)


# In[40]:


not_in_both = [x for x in minimal_training_data + non_minimal_training_data if x not in minimal_training_data or x not in non_minimal_training_data]


# In[41]:


print(not_in_both)


# In[42]:


# Now I want to split the data in two.  


random.shuffle(minimal_training_data)

first_half_minimal_training_data=(minimal_training_data[:5000])
second_half_minimal_training_data=(minimal_training_data[5000:])


len(first_half_minimal_training_data)

len(second_half_minimal_training_data)


complexity_one_data=[] 

for words in first_half_minimal_training_data: 
    complexity_one=words_getting_bigger(words) 
    complexity_one_data.append(complexity_one)


len(complexity_one_data)

complexity_one_training_data_labelled=[] 

for words in complexity_one_data: 
    labelled=(words,0) 
    complexity_one_training_data_labelled.append(labelled)

minimal_training_data_labelled=[] 

for words in second_half_minimal_training_data: 
        label=(words,1) 
        minimal_training_data_labelled.append(label)

training_data=complexity_one_training_data_labelled+minimal_training_data_labelled


# In[43]:


len(training_data)


# # Importing my Feature Vector Functions

# In[44]:


# Welcome to the 06/11 version of NK's Pattern Recognition code. Hopefully this will be the last version.

# # 0. Initial Useful Functions

# In[1]:


# This is a useful function for unittesting
def generate_random_word(length, generators):
    word = []
    for _ in range(length):
        generator = random.choice(generators)
        power = random.choice([1, -1])
        word.append((generator, power))
    return word 

y=generate_random_word(100,("a","b")) 
print(y) 


# In[2]:


# (Function #1) Reduces and normalises words: 

def reduce_and_normalise(word):
    """
    Reduces and normalizes a word represented as a list of tuples.
    Returns the reduced and normalized word.
    """
    reduced = []
    
    for generator, power in word:
        if reduced and reduced[-1][0] == generator:
            combined_power = reduced[-1][1] + power
            
            if combined_power != 0:
                reduced[-1] = (generator, combined_power)
            else:
                reduced.pop()
        else:
            reduced.append((generator, power))
    
    normalized = []
    
    for gen, power in reduced:
        if power < 0:
            # If power is negative, convert it to -1 by repeating the generator
            normalized.extend([(gen, -1)] * abs(power))
        elif power > 0:
            # If power is positive, convert it to 1 by repeating the generator
            normalized.extend([(gen, 1)] * power)
        else:
            # If power is zero, ignore it
            pass
    
    return normalized


# Test and Example Usage

# In[3]:




class TestNotebook(unittest.TestCase):
    
    def test_reduce_word(self):
        word0=[] 
        result = reduce_and_normalise(word0) 
        self.assertEqual(result, [])
    
    def test_single_generator(self):
        word1 = [("a", 1)]
        result = reduce_and_normalise(word1)
        self.assertEqual(result, [("a", 1)]) 
    
    def test_canceling_generators(self):
        word2 = [("a", 1), ("a", -1), ("b", 2), ("b", -2)] 
        result = reduce_and_normalise(word2)
        self.assertEqual(result, [])

unittest.main(argv=[''], verbosity=2, exit=False)


# In[4]:


#Example Usage 
w= [("a",0),("a",-4)]
word = [('a', 1), ('b', 1), ('b', 1), ('a', -1), ('b', 1), ('a', -1),('a', 1), ('b', 1), ('b', 1), ('a', -1), ('b', 1), ('a', -1)]

reduce_and_normalise(word) 


# In[5]:


# (Function #2) Reduced word length 

def red_word_length(word):
    
    # Reduce the word to condensed tuple form 
    reduced =  reduce_and_normalise(word)
    
    # Calculate the sum of absolute values of the second part of each tuple
    result = sum(abs(w[1]) for w in reduced) 
    return result


# # 0.1 Test and Example Usage

# In[6]:


class TestNotebook(unittest.TestCase):
    
    def test_empty(self):
        word0=[] 
        result = red_word_length(word0) 
        self.assertEqual(result, 0)
    
    def single_length(self):
        word1 = [("a", 1)]
        result = red_word_length(word1)
        self.assertEqual(result, 1) 
    
    def test_cancelling_generators(self):
        w= [("a",10),("a",-4),("b",100),("b",-4)]
        result = red_word_length(w)
        self.assertEqual(result, 102)

unittest.main(argv=[''], verbosity=2, exit=False)


# In[7]:


w= [("a",10),("a",-4),("b",10),("b",-4)]
print(red_word_length(w))


# # 1. Generating Lists

# We need some functions that generate possible lists of words from other lists of words in a specified way. These lists need to be in reduced and in normalised form. 

# In[8]:


# (Functions #3 and #4) # 3 Generates lists in the form sepcified in HMM, and #4 reduces and normalises these lists. 


def generate_lists(V, U): # Need to make V and U have the same length - and thus 
    result = [[]]

    for index in range(len(V)):
        new_result = []
        for word in U[index]:
            for combo in result:
                new_result.append(combo + [word, V[index]])
        result = new_result
        
    combinations = []
    
    for data in result: 
        concise_list =[tup for sublist in data if sublist for tup in sublist]
        combinations.append(concise_list)
    
    return combinations

# Now: could define v_{k+1} to be the empty word 

def generate_norm_reduced_lists(V, U): # We want to work with this function
    big_list = []
    unique_list = []
    for lists in generate_lists(V, U):
        reduced = reduce_and_normalise(lists)
        big_list.append(reduced)
        
        
        

        for lists in big_list:
            if lists not in unique_list:
                unique_list.append(lists)# remove duplicates
    return unique_list


# Some example usage and unit tests on generate lists

# In[9]:


# V should have k elements, which are words
V = [
    [('a', 1),('a',1)],
    [('b', -1)],
    []
]

# U should have k+1 elements, which are lists of words
U = [
    [[]],
    [[]],
    [[('a', -1)],[('b', -1)]]
]

print(generate_norm_reduced_lists(V,U))


# In[10]:


V = [[("a", 1)],[]] 
U = [
[[]],[[]]
] 

generate_lists(V,U)


# # 1.1 Unit Tests

# In[11]:


# This function could be made better by including normalise and reduce to words. Actually no need.  

class TestNotebook(unittest.TestCase):
    
    def test_empty_list_1(self): 
        V = [[("a", 1)],[]] 
        U = [
        [[]],[[]]
        ]
        result = generate_lists(V, U) 
        self.assertEqual(result, [[("a", 1)]])
     
    def test_big_list_1(self): 
        V = [
            [("a", 1), ("a", 1)],
            [("a", 1), ("b", 1)],
            []
        ] 
        U = [
            [ [("a", 1), ("a", 1)], [("b", 1), ("a", 1)] ],
            [ [("a", -1), ("b", 1)], [("b", 1), ("a", 1)] ],
            [ [("a", 1), ("b", 1)], [("b", 1), ("a", -1)] ] 
        ] 
        
        result = generate_lists(V, U)  
        

        exp_result = [ 
        [("a",1),("a",1),("a",1),("a",1),("a",-1),("b",1),("a",1),("b",1),("a",1),("b",1)], 
        [("a",1),("a",1),("a",1),("a",1),("b",1),("a",1),("a",1),("b",1),("a",1),("b",1)],
        [("a",1),("a",1),("a",1),("a",1),("a",-1),("b",1),("a",1),("b",1),("b",1),("a",-1)], 
        [("a",1),("a",1),("a",1),("a",1),("b",1),("a",1),("a",1),("b",1),("b",1),("a",-1)], 
        [("b",1),("a",1),("a",1),("a",1),("a",-1),("b",1),("a",1),("b",1),("a",1),("b",1)],
        [("b",1),("a",1),("a",1),("a",1),("b",1),("a",1),("a",1),("b",1),("a",1),("b",1)],
        [("b",1),("a",1),("a",1),("a",1),("b",1),("a",1),("a",1),("b",1),("b",1),("a",-1)], 
        [("b",1),("a",1),("a",1),("a",1),("a",-1),("b",1),("a",1),("b",1),("b",1),("a",-1)]
        ]
          
        shared_list=[] 
        for x1 in result: 
            for x2 in exp_result: 
                if x1==x2: 
                    shared_list.append(x1)

        self.assertEqual(2*len(shared_list),len(result)+len(exp_result)) # How to check the lists are equal up to reordering.

    def test_cancelling_list_1(self): 
        V = [
        [("a", 1), ("a", 1)],
        [("a", 1), ("b", 1)],
        [] 
        ] 
        U = [
        [ [("a", 1)] ],
        [ [("a", -1)] ],
        [ [("a", 1)] ] 
        ] 
        expected_result = [
        [("a", 1), ("a", 1), ("a", 1), ("b", 1), ("a", 1)]
        ]
    
        result = generate_norm_reduced_lists(V, U) 
        
        self.assertEqual(result, expected_result)

unittest.main(argv=[''], verbosity=2, exit=False)


# # 2. Counting Functions

# In[12]:


# (Function #5 and #6) Counts occurrences of one word in another,  
def count_occurrences(w, v):
    if not v:  # Check if v is empty
        return 0

    w_len, v_len = len(w), len(v)

    count = 0

    i = 0
    while i <= w_len - v_len:
        match = True
        for j in range(v_len):
            if w[i + j] != v[j]:
                match = False
                break

        if match:
            count += 1
        i += 1

    return count


def count_occurrences_in_list(w, U):
    total_count = 0
    
    for v in U:
        count_v = count_occurrences(w, v)
        total_count += count_v
    
    return total_count


# # 2.1 Example Usage and Unit Testing

# In[13]:


w = "abracadabra"
v = ""
count = count_occurrences(w, v)
print(f"The substring '{v}' appears {count} times in '{w}'.")


# In[14]:


w = "abracadabra"
U = ["abra", "cad", "dabra", "abc"]
total_count = count_occurrences_in_list(w, U)
print(f"The total count of substrings in the list in '{w}' is {total_count}.") 


# In[ ]:





# In[15]:


w = [('a', 1), ('b', 1), ('a', 1), ('b', 1), ('a', 1), ('a', 1), ('b', -1), ('a', 1), ('b', 1), ('a', -1)]
v = [('a', 1), ('b', 1), ('a', 1)]
count = count_occurrences(w, v)


print(f"The substring '{v}' appears {count} times in '{w}'.")
# Output: The substring '[('a', 1), ('b', 1), ('a', 1)]' appears 3 times in '[('a', 1), ('b', 1), ('a', 1), ('b', 1), ('a', 1), ('a', 1), ('b', -1), ('a', 1), ('b', 1), ('a', -1)]'


# In[16]:


w = [('a', 1), ('b', 1), ('a', 1), ('b', 1), ('a', 1), ('a', 1), ('b', -1), ('a', 1), ('b', 1), ('a', -1)]
U = [[('a', 1), ('b', 1), ('a', 1)], [('a', 1), ('b', 1)], [('b', 1), ('a', -1), ('a', 1)]]
total_count = count_occurrences_in_list(w, U)
print(f"The total count of substrings in the list in '{w}' is {total_count}.")


# In[17]:


class TestNotebook(unittest.TestCase):
    
    def test_occurences_simple(self): 
        w=[("a",1)] 
        v=[("a",1)]
        self.assertEqual(count_occurrences(w, v), 1) 
        
    def test_occurences_empty(self): 
        w=[("a",1)] 
        v=[]
        self.assertEqual(count_occurrences(w, v), 0)
        
    def test_overlapping_occurences(self): 
        new_w=[("a",1),("b",1),("a",1),("b",1),("a",1),("b",1),("a",1),("b",1),("a",1),("b",1),]
        new_v=[("a",1),("b",1),("a",1)] 
        self.assertEqual(count_occurrences(new_w, new_v), 4) 
        
    def test_occurrences_in_list_empty(self): 
        w=[("a",1)] 
        U=[[("a,1"),("b",1)],[]] 
        self.assertEqual(count_occurrences_in_list(w,U),0)
        
    def test_occurrences_in_list_simple(self): # Beware of duplicates
        w=[("a",1)] 
        U=[[("a",1),("b",1)],[("a",1)],[("a",1)]] 
        self.assertEqual(count_occurrences_in_list(w,U),2)
        
    def test_overlapping_occurences_in_list(self): 
        w=[("a",1),("b",1),("a",1),("b",1),("a",1),("b",1),("a",1),("b",1),("a",1),("b",1),]
        U=[[("a",1),("b",1),("a",1)],[("b",1),("a",1),("b",1)],[("a",1),("b",1),("a",1),("b",1)],[("b",1),("a",1),("b",1),("a",1)]]
        self.assertEqual(count_occurrences_in_list(w, U), 15)
        
    unittest.main(argv=[''], verbosity=2, exit=False)


# In[ ]:


# In[45]:


# # 3. Feature Vectors.

# # 3.1 Defining $U_i$ and $W_i$

# In[18]:


def U(n):
    if n == 0:
        return [[]]
    elif n == 1:
        return [[('a', 1)], [('a', -1)], [('b', 1)], [('b', -1)]]
    else:
        U_n_minus_1 = U(n - 1)
        Xplusminus = [('a', 1), ('a', -1), ('b', 1), ('b', -1)]
        U_n = []

        for sublist in U_n_minus_1:
            for item in Xplusminus:
                last_item = sublist[-1] if sublist else None

                cancels = last_item is not None and (last_item[0] == item[0] and last_item[1] == -item[1])

                if not cancels:
                    new_sublist = sublist + [item]
                    U_n.append(new_sublist)

        return U_n


# In[19]:


print(U(1)) 
print(U(2))


# In[ ]:


# In[46]:


# In[20]:


def W(n): 
    
    m=n+1
    
    W=[]
    
    for i in range(m): 
        for words in U(i):
            W.append(words) 
    return W 


# In[47]:


# In[21]:


W(1)


# In[48]:


# # 3.1.1 Unit Testing for U_i and W_i

# In[22]:


class TestNotebook(unittest.TestCase):
    
    def test_U_n_length(self): #Testing that we have the right lengths
        for n in range(1, 10):
            self.assertEqual(len(U(n)), 4 * (3 ** (n - 1))) 
            
    def test_W_n_length(self): # Failed test first time, now ammended
        x=len(W(1)) 
        y=len(U(0))+len(U(1))
        self.assertEqual(x,y)

unittest.main(argv=[''], verbosity=2, exit=False)


# # 3.2.  $f_0(w)=\frac{1}{|w|} \langle C(w,a) : a \in X^{\pm1} \rangle.$

# In[ ]:


# In[49]:


# In[23]:

def f_0(word): 
    
    reduced = reduce_and_normalise(word) 
    
    count_list = [] 
    
    Xplusminus = [('a', 1), ('a', -1), ('b', 1), ('b', -1)] 
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    for item in list_Xplusminus:
        count_list.append((count_occurrences(reduced, item))/red_word_length(word))
    return np.array(count_list)


# In[24]:


word = [('a', 1), ('b', -1), ('a', 1), ('a', -1), ('b', 1), ('b', -1)]
result = f_0(word)
print(result) 

type(f_0([("a",-1)])) 
result=f_0([("a",1),("b",1)]) 
print(result)


# In[50]:


# # 3.2.1 Test for $f_0$

# In[25]:


class TestNotebook(unittest.TestCase):

    def test_f_0_length_1(self):
        result = f_0([("a", 1)]) + f_0([("a", -1)]) + f_0([("b", 1)]) + f_0([("b", -1)])
        exp_result = np.array([1, 1, 1, 1])
        v = np.array_equal(exp_result, result)
        self.assertTrue(v)

    def test_f_0_length_1_cancel(self):
        result = f_0([("a", 1),("a",1),("a",-1)])
        exp_result = np.array([1, 0, 0, 0])
        v = np.array_equal(exp_result, result)
        self.assertTrue(v) 
        
    def test_f_0_length_2(self):
        result = f_0([("a", 1),("b",1)])
        exp_result = np.array([0.5, 0, 0.5, 0])
        v = np.array_equal(exp_result, result)
        self.assertTrue(v)
    
    
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)


# # 3.3.  $ f_1(w) = \frac{1}{|w|} < C(w,a) : |a|=2 \} > $

# In[26]:


def f_1(word): 
    
    reduced = reduce_and_normalise(word)
    
    count_list = [] 
    
    for item in U(2):
        count_list.append((count_occurrences(reduced , item))/red_word_length(word))
        
    return np.array(count_list) 


# In[51]:


# # 3.3.1 Unit Test for f_2

# In[27]:


class TestNotebook(unittest.TestCase):

    def test_f_01_cancel(self):
        result = f_1([("a", 1),("a",-1),("a",1)])
        exp_result = f_1([("a",1)])
        v = np.array_equal(exp_result, result)
        self.assertTrue(v)
    
    def test_f_01_simple(self): 
        result =  f_1([("a",1),("a",1)])
        exp_result = np.array([0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]) 
        v = np.array_equal(exp_result, result)
        self.assertTrue(v) 
        
    def test_f_01_large(self):
        result =  f_1([("a",100)])
        exp_result = np.array([0.99, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]) 
        v = np.array_equal(exp_result, result)
        self.assertTrue(v)
    
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)


# In[52]:


# # 3.4 $$ f_2(w) = \frac{1}{|w|} \langle C(w,x_1U_1x_2) : x_1,x_2 \in X^{\pm1} \rangle  $$

# In[28]:


def f_2(word):
    
    def U(n): # U_i and W_i defined in the function to make them easier to call
        if n == 0:
            return []
        elif n == 1:
            return [[('a', 1)], [('a', -1)], [('b', 1)], [('b', -1)]]
        else:
            U_n_minus_1 = U(n - 1)
            Xplusminus = [('a', 1), ('a', -1), ('b', 1), ('b', -1)]
            U_n = []

            for sublist in U_n_minus_1:
                for item in Xplusminus:
                    last_item = sublist[-1] if sublist else None

                    cancels = last_item is not None and (last_item[0] == item[0] and last_item[1] == -item[1])

                    if not cancels:
                        new_sublist = sublist + [item]
                        U_n.append(new_sublist)

            return U_n 

    def W(n): 
        m = n + 1
        W = [[]]

        for i in range(m): 
            for words in U(i):
                W.append(words) 
        return W 

    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        U(1),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus:
        for x2 in Xplusminus:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, U_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count) 


# In[53]:


# An alternative function that uses an alternative list generating method.

# In[29]:


def list_concat(x_1,x_2,U): 
    
    new_words=[]
    for word in U: 
        word.insert(0,x_1) 
        word.append(x_2)  
        new_words.append(reduce_and_normalise(word)) 
    return new_words

def F_2(word):
    
    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]

    combos=[]

    for x_1 in Xplusminus: 
        for x_2 in Xplusminus: 
            combinations=list_concat(x_1,x_2,U(1)) 
        
            combos.append(combinations)
         
    
    count=[]
    
    for V in combos: 
        x=count_occurrences_in_list(reduced_word,V)/ red_word_length(reduced_word) 
        count.append(x)
        
        
    return np.array(count)  


# In[54]:


# In[30]:




start = time.time()
input_word = [("a",10),("b",10),("a",-1)]

print(F_2(input_word))
end = time.time()
print(end - start)

start = time.time()
input_word = [("a",10),("b",10),("a",-1)]

print(f_2(input_word))
end = time.time()
print(end - start)
 # both output same values.  


# In[55]:


# # 3.4.1 Unit testing

# In[31]:


class TestNotebook(unittest.TestCase):

    def test_f_02_cancel(self):
        result = f_2([("a", 1),("a",-1),("a",1)])
        exp_result = f_2([("a",1)])
        v = np.array_equal(exp_result, result)
        self.assertTrue(v)
    
    def test_f_02_simple(self): 
        result =  f_2([("a",1),("a",1)])
        exp_result = np.array([1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.]) 
        v = np.array_equal(exp_result, result)
        self.assertTrue(v) 
        
    def test_f_02_large(self):
        result =  f_2([("a",100)]) #Note that the values in the array are bounded by 2. 
        exp_result = np.array([1.98, 1.  , 1.  , 1.  , 1.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  ,
       0.  , 1.  , 0.  , 0.  , 0.  ]) 
        v = np.array_equal(exp_result, result)
        self.assertTrue(v)
    
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)


# In[56]:


# In[32]:


f_2([("a",100)])


# # 3.5 Feature Vector 4: $$ f_3(w) = \frac{1}{|w|} < C(w,x_1U_2x_2) : x_1,x_2 \in X^{\pm1} \} > $$

# In[33]:


# In[57]:


x=2 
y=3


# In[58]:


x+y


# In[59]:


def f_3(word): #Almost identical code to f_2.
    
    def U(n): # U_i and W_i defined in the function to make them easier to call
        if n == 0:
            return []
        elif n == 1:
            return [[('a', 1)], [('a', -1)], [('b', 1)], [('b', -1)]]
        else:
            U_n_minus_1 = U(n - 1)
            Xplusminus = [('a', 1), ('a', -1), ('b', 1), ('b', -1)]
            U_n = []

            for sublist in U_n_minus_1:
                for item in Xplusminus:
                    last_item = sublist[-1] if sublist else None

                    cancels = last_item is not None and (last_item[0] == item[0] and last_item[1] == -item[1])

                    if not cancels:
                        new_sublist = sublist + [item]
                        U_n.append(new_sublist)

            return U_n 

    def W(n): 
        m = n + 1
        W = [[]]

        for i in range(m): 
            for words in U(i):
                W.append(words) 
        return W 

    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        U(2),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus:
        for x2 in Xplusminus:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, U_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count) 


# In[60]:


# In[34]:


def F_3(word): # Alternative function
    
    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]

    combos=[]

    for x_1 in Xplusminus: 
        for x_2 in Xplusminus: 
            combinations=list_concat(x_1,x_2,U(2)) 
        
            combos.append(combinations)
         
    count=[]
    
    for V in combos: 
        x=count_occurrences_in_list(reduced_word,V)/ red_word_length(reduced_word) 
        count.append(x)
        
        
    return np.array(count)  


# In[61]:


# # 3.5.1 Unit Testing

# In[35]:


class TestNotebook(unittest.TestCase):
    
    def test_f_3_and_F_3_equality(self):
        x = [[("a", 3), ("b", -2)], [("a", -1), ("b", 2)], [("a", 3), ("b", -2), ("a", 2)]]
        for y in x:
            result = np.array_equal(f_3(y), F_3(y))
            self.assertTrue(result) 
    
    def test_f_3_simple(self):  
        y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        result=np.array_equal(f_3([("a",1)]),y) 
        self.assertTrue(result)  
    
    def test_large_first_value(self): 
        y=f_3([("a",100)])[0] 
        self.assertTrue(y,0.97)
        
            
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)


# In[ ]:


# # 
# 
# # # 3.6: $$ f_4(w) = \frac{1}{|w|} < C(w,x_1U_3x_2) : x_1,x_2 \in X^{\pm1} \} > $$

# In[62]:


# In[36]:


def f_4(word):
    
    def U(n):
        if n == 0:
            return []
        elif n == 1:
            return [[('a', 1)], [('a', -1)], [('b', 1)], [('b', -1)]]
        else:
            U_n_minus_1 = U(n - 1)
            Xplusminus = [('a', 1), ('a', -1), ('b', 1), ('b', -1)]
            U_n = []

            for sublist in U_n_minus_1:
                for item in Xplusminus:
                    last_item = sublist[-1] if sublist else None

                    cancels = last_item is not None and (last_item[0] == item[0] and last_item[1] == -item[1])

                    if not cancels:
                        new_sublist = sublist + [item]
                        U_n.append(new_sublist)

            return U_n 

    def W(n): 
        m = n + 1

        W = [[]]

        for i in range(m): 
            for words in U(i):
                W.append(words) 
        return W 

    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        U(3),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus:
        for x2 in Xplusminus:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, U_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count)


# In[63]:


# In[37]:


def list_concat(x_1,x_2,U): # Alternative function again.
    
    new_words=[]
    for word in U: 
        word.insert(0,x_1) 
        word.append(x_2)  
        new_words.append(reduce_and_normalise(word)) 
    return new_words

def F_4(word): # Alternative function
    
    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]

    combos=[]

    for x_1 in Xplusminus: 
        for x_2 in Xplusminus: 
            combinations=list_concat(x_1,x_2,U(3)) 
        
            combos.append(combinations)
         
    count=[]
    
    for V in combos: 
        x=count_occurrences_in_list(reduced_word,V)/ red_word_length(reduced_word) 
        count.append(x)
        
        
    return np.array(count) 


# In[ ]:


# In[64]:


# # 3.6.1 Unit Testing for $f_4$

# In[38]:


class TestNotebook(unittest.TestCase):
    
    def test_f_4_and_F_4_equality(self):
        x = [[("a", 3), ("b", -2)], [("a", -1), ("b", 2)], [("a", 3), ("b", -2), ("a", 2)]]
        for y in x:
            result = np.array_equal(f_4(y), F_4(y))
            self.assertTrue(result) 
    
    def test_f_4_simple(self):  
        y=[0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1.]
        result=np.array_equal(f_4([("a",1)]),y) 
        self.assertTrue(result)  
    
    def test_f_4_large_first_value(self): 
        y=f_4([("a",100)])[0] 
        self.assertTrue(y,0.96)
        
    def test_f_4_random(self): 
        y=generate_random_word(100,("a","b")) 
        result=np.array_equal(f_4(y), F_4(y)) 
        self.assertTrue(y)
        
            
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)


# In[65]:


# # 3.7: $$ f_5(w) = \frac{1}{|w|} < C(w,x_1W_1x_2) : x_1,x_2 \in X^{\pm1} \} > $$


# In[66]:


# In[39]:


def f_5(word):
    
    def U(n):
        if n == 0:
            return []
        elif n == 1:
            return [[('a', 1)], [('a', -1)], [('b', 1)], [('b', -1)]]
        else:
            U_n_minus_1 = U(n - 1)
            Xplusminus = [('a', 1), ('a', -1), ('b', 1), ('b', -1)]
            U_n = []

            for sublist in U_n_minus_1:
                for item in Xplusminus:
                    last_item = sublist[-1] if sublist else None

                    cancels = last_item is not None and (last_item[0] == item[0] and last_item[1] == -item[1])

                    if not cancels:
                        new_sublist = sublist + [item]
                        U_n.append(new_sublist)

            return U_n 

    def W(n): 
        m = n + 1
        W = [[]]

        for i in range(m): 
            for words in U(i):
                W.append(words) 
        return W 

    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        W(1),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus:
        for x2 in Xplusminus:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, U_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count)


# In[67]:


# In[40]:


def list_concat(x_1,x_2,U): # Alternative function again.
    
    new_words=[]
    for word in U: 
        word.insert(0,x_1) 
        word.append(x_2)  
        new_words.append(reduce_and_normalise(word)) 
    return new_words

def F_5(word): # Alternative function
    
    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]

    combos=[]

    for x_1 in Xplusminus: 
        for x_2 in Xplusminus: 
            combinations=list_concat(x_1,x_2,W(1)) 
        
            combos.append(combinations)
         
    count=[]
    
    for V in combos: 
        x=count_occurrences_in_list(reduced_word,V)/ red_word_length(reduced_word) 
        count.append(x)
        
        
    return np.array(count) 


# In[41]:


f_5([("a",10000)])


# In[68]:


# # 3.7.1 Unit Testing for $f_5$

# In[42]:


class TestNotebook(unittest.TestCase):
    
    def test_f_5_and_F_5_equality(self):
        x = [[("a", 3), ("b", -2)], [("a", -1), ("b", 2)], [("a", 3), ("b", -2), ("a", 2)]]
        for y in x:
            result = np.array_equal(f_5(y), F_5(y))
            self.assertTrue(result) 
    
    def test_f_5_simple(self):  
        y=[1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.]
        result=np.array_equal(f_5([("a",1)]),y) 
        self.assertTrue(result)  
    
    def test_f_5_large_first_value(self): # This is a faulty test
        y=f_5([("a",1)])[0] 
        self.assertTrue(y,0.96)
        
    def test_f_5_random(self): 
        y=generate_random_word(100,("a","b")) 
        result=np.array_equal(f_5(y), F_5(y)) 
        self.assertTrue(y)
        
            
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)


# # # 3.8: $$ f_6(w) = \frac{1}{|w|} < C(w,x_1W_3x_2) : x_1,x_2 \in X^{\pm1} \} > $$
# 
# # In[ ]:

# In[69]:


# In[43]:


def f_6(word):
    
    def U(n):
        if n == 0:
            return []
        elif n == 1:
            return [[('a', 1)], [('a', -1)], [('b', 1)], [('b', -1)]]
        else:
            U_n_minus_1 = U(n - 1)
            Xplusminus = [('a', 1), ('a', -1), ('b', 1), ('b', -1)]
            U_n = []

            for sublist in U_n_minus_1:
                for item in Xplusminus:
                    last_item = sublist[-1] if sublist else None

                    cancels = last_item is not None and (last_item[0] == item[0] and last_item[1] == -item[1])

                    if not cancels:
                        new_sublist = sublist + [item]
                        U_n.append(new_sublist)

            return U_n 

    def W(n): 
        m = n + 1
        W = [[]]

        for i in range(m): 
            for words in U(i):
                W.append(words) 
        return W 

    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        W(3),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus:
        for x2 in Xplusminus:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, U_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count) 


# In[70]:


# In[44]:


def list_concat(x_1,x_2,U): # Alternative function again.
    
    new_words=[]
    for word in U: 
        word.insert(0,x_1) 
        word.append(x_2)  
        new_words.append(reduce_and_normalise(word)) 
    return new_words

def F_6(word): # Alternative function
    
    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = [("a", 1), ("a", -1), ("b", 1), ("b", -1)]

    combos=[]

    for x_1 in Xplusminus: 
        for x_2 in Xplusminus: 
            combinations=list_concat(x_1,x_2,W(3)) 
        
            combos.append(combinations)
         
    count=[]
    
    for V in combos: 
        x=count_occurrences_in_list(reduced_word,V)/ red_word_length(reduced_word) 
        count.append(x)
        
        
    return np.array(count) 


# In[71]:


# # 3.8.1 Unit Testing for $f_6$

# In[45]:


class TestNotebook(unittest.TestCase):
    
    def test_f_6_and_F_6_equality(self):
        x = [[("a", 3), ("b", -2)], [("a", -1), ("b", 2)], [("a", 3), ("b", -2), ("a", 2)]]
        for y in x:
            result = np.array_equal(f_6(y), F_6(y))
            self.assertTrue(result) 
    
    def test_f_6_simple(self):  
        y=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        result=np.array_equal(f_6([("a",1)]),y) 
        self.assertTrue(result)  
    
    def test_f_5_large_first_value(self): # This is a faulty test
        y=f_6([("a",1)])[0] 
        self.assertTrue(y,0.96)
        
    def test_f_5_random(self): 
        y=generate_random_word(100,("a","b")) 
        result=np.array_equal(f_6(y), F_6(y)) 
        self.assertTrue(y)
        
            
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)


# # Test Data

# In[72]:


non_minimal_test_data = [] # this could be made faster

for length in range(1, 1001):
    words = list_of_cyclic_reduced_words(10, length)
    non_minimal_test_data.extend(words)  
    
minimal_test_data=[find_minimal_rep(words) for words in non_minimal_test_data]


# In[73]:


random.shuffle(minimal_test_data)

first_half_minimal_test_data=(minimal_test_data[:5000])
second_half_minimal_test_data=(minimal_test_data[5000:])


# In[74]:


complexity_one_test_data=[words_getting_bigger(words) for words in first_half_minimal_test_data]


# In[75]:


complexity_one_test_data_labelled=[(word,0) for word in complexity_one_test_data]


# In[76]:


minimal_test_data_labelled=[(word,1) for word in second_half_minimal_test_data]    


# In[77]:


test_data=complexity_one_test_data_labelled+minimal_test_data_labelled


# # Making the matrices

# First we split our training data into labels and words

# In[78]:


training_data_words = [t[0] for t in training_data] 
training_data_labels=[t[1] for t in training_data]  


# In[79]:


test_data_words = [t[0] for t in test_data] 
test_data_labels=[t[1] for t in test_data]  


# # $f_1$ matrices

# In[ ]:


f_0_train_arrays=[f_0(t) for t in training_data_words ] # Making the f_0 matrix 

f_0_train_arrays_non_min=f_0_train_arrays[:5000] # Min and non-min arrays
f_0_train_arrays_min=f_0_train_arrays[5000:]


# In[320]:


V0_train=np.row_stack(f_0_train_arrays)  

V0_non_min_train=np.row_stack(f_0_train_arrays_non_min) 
V0_min_train=np.row_stack(f_0_train_arrays_min) 


# In[81]:


f_1_train_arrays=[f_1(t) for t in training_data_words ] # Making the f_1 matrix 

f_1_train_arrays_non_min=f_1_train_arrays[:5000] # Min and non-min arrays
f_1_train_arrays_min=f_1_train_arrays[5000:]

V1_train=np.row_stack(f_1_train_arrays)  

V1_non_min_train=np.row_stack(f_1_train_arrays_non_min) 
V1_min_train=np.row_stack(f_1_train_arrays_min) 


# In[82]:


f_2_train_arrays=[f_2(t) for t in training_data_words] 
V2_train=np.row_stack(f_2_train_arrays) 

f_2_train_arrays_non_min=f_2_train_arrays[:5000] # Min and non-min arrays
f_2_train_arrays_min=f_2_train_arrays[5000:]

V2_train=np.row_stack(f_2_train_arrays)  

V2_non_min_train=np.row_stack(f_2_train_arrays_non_min) 
V2_min_train=np.row_stack(f_2_train_arrays_min) 


# In[83]:


f_3_train_arrays=[f_3(t) for t in training_data_words]
V3_train=np.row_stack(f_3_train_arrays)

f_3_train_arrays_non_min=f_3_train_arrays[:5000] # Min and non-min arrays
f_3_train_arrays_min=f_3_train_arrays[5000:]

V3_train=np.row_stack(f_3_train_arrays)  

V3_non_min_train=np.row_stack(f_3_train_arrays_non_min) 
V3_min_train=np.row_stack(f_3_train_arrays_min) 


# In[84]:


f_4_train_arrays=[f_4(t) for t in training_data_words]
V4_train=np.row_stack(f_4_train_arrays)

f_4_train_arrays_non_min=f_4_train_arrays[:5000] # Min and non-min arrays
f_4_train_arrays_min=f_4_train_arrays[5000:]

V4_train=np.row_stack(f_4_train_arrays)  

V4_non_min_train=np.row_stack(f_4_train_arrays_non_min) 
V4_min_train=np.row_stack(f_4_train_arrays_min) 


# In[85]:


f_5_train_arrays=[f_5(t) for t in training_data_words]
V5_train=np.row_stack(f_5_train_arrays)

f_5_train_arrays_non_min=f_5_train_arrays[:5000] # Min and non-min arrays
f_5_train_arrays_min=f_5_train_arrays[5000:]

V5_train=np.row_stack(f_5_train_arrays)  

V5_non_min_train=np.row_stack(f_5_train_arrays_non_min) 
V5_min_train=np.row_stack(f_5_train_arrays_min) 


# In[86]:


f_6_train_arrays=[f_6(t) for t in training_data_words]
V6_train=np.row_stack(f_6_train_arrays)

f_6_train_arrays_non_min=f_6_train_arrays[:5000] # Min and non-min arrays
f_6_train_arrays_min=f_6_train_arrays[5000:]

V6_train=np.row_stack(f_6_train_arrays)  

V6_non_min_train=np.row_stack(f_6_train_arrays_non_min) 
V6_min_train=np.row_stack(f_6_train_arrays_min) 


# In[317]:


P_train=np.array(training_data_labels) # Making probability vector.   
P_train.shape


# In[88]:


# Now for the test data


# In[ ]:


f_0_test_arrays=[f_0(t) for t in test_data_words ] # Making the f_0 matrix 

f_0_test_arrays_non_min=f_0_test_arrays[:5000] # Min and non-min arrays
f_0_test_arrays_min=f_0_test_arrays[5000:]


# In[319]:


V0_test=np.row_stack(f_0_test_arrays)  

V0_non_min_test=np.row_stack(f_0_test_arrays_non_min) 
V0_min_test=np.row_stack(f_0_test_arrays_min) 


# In[316]:


f_1_test_arrays=[f_1(t) for t in test_data_words ] # Making the f_1 matrix 

f_1_test_arrays_non_min=f_1_test_arrays[:5000] # Min and non-min arrays
f_1_test_arrays_min=f_1_test_arrays[5000:]

V1_test=np.row_stack(f_1_test_arrays)  

V1_non_min_test=np.row_stack(f_1_test_arrays_non_min) 
V1_min_test=np.row_stack(f_1_test_arrays_min) 


# In[91]:


f_2_test_arrays=[f_2(t) for t in test_data_words ] # Making the f_2 matrix 

f_2_test_arrays_non_min=f_2_test_arrays[:5000] # Min and non-min arrays
f_2_test_arrays_min=f_2_test_arrays[5000:]

V2_test=np.row_stack(f_2_test_arrays)  

V2_non_min_test=np.row_stack(f_2_test_arrays_non_min) 
V2_min_test=np.row_stack(f_2_test_arrays_min) 


# In[92]:


f_3_test_arrays=[f_3(t) for t in test_data_words ] # Making the f_3 matrix 

f_3_test_arrays_non_min=f_3_test_arrays[:5000] # Min and non-min arrays
f_3_test_arrays_min=f_3_test_arrays[5000:]

V3_test=np.row_stack(f_3_test_arrays)  

V3_non_min_test=np.row_stack(f_3_test_arrays_non_min) 
V3_min_test=np.row_stack(f_3_test_arrays_min) 


# In[93]:


f_4_test_arrays=[f_4(t) for t in test_data_words ] # Making the f_4 matrix 

f_4_test_arrays_non_min=f_4_test_arrays[:5000] # Min and non-min arrays
f_4_test_arrays_min=f_4_test_arrays[5000:]

V4_test=np.row_stack(f_4_test_arrays)  

V4_non_min_test=np.row_stack(f_4_test_arrays_non_min) 
V4_min_test=np.row_stack(f_4_test_arrays_min)  


# In[94]:


f_5_test_arrays=[f_5(t) for t in test_data_words ] # Making the f_5 matrix 

f_5_test_arrays_non_min=f_5_test_arrays[:5000] # Min and non-min arrays
f_5_test_arrays_min=f_5_test_arrays[5000:]

V5_test=np.row_stack(f_5_test_arrays)  

V5_non_min_test=np.row_stack(f_5_test_arrays_non_min) 
V5_min_test=np.row_stack(f_5_test_arrays_min) 


# In[95]:


f_6_test_arrays=[f_6(t) for t in test_data_words ] # Making the f_6 matrix 

f_6_test_arrays_non_min=f_6_test_arrays[:5000] # Min and non-min arrays
f_6_test_arrays_min=f_6_test_arrays[5000:]

V6_test=np.row_stack(f_6_test_arrays)  

V6_non_min_test=np.row_stack(f_6_test_arrays_non_min) 
V6_min_test=np.row_stack(f_6_test_arrays_min) 


# In[96]:


V6_test=np.row_stack(f_6_test_arrays)  

V6_non_min_test=np.row_stack(f_6_test_arrays_non_min) 
V6_min_test=np.row_stack(f_6_test_arrays_min)


# In[315]:


P_test=np.array(test_data_labels) # Making probability vector.   
P_test.shape


# In[ ]:





# In[ ]:





# # Neural Network

# In[322]:


# Define the number of runs
num_runs = 10
mse_scores = []

# Copy the data for train-test split outside the loop
V0_train_copy = V0_train.copy()
P_train_copy = P_train.copy()

# Train-test split
V0_train, V0_val, P_train, P_val = train_test_split(V0_train_copy, P_train_copy, test_size=0.1, random_state=42)

for run in range(num_runs):
    # Model architecture
    model = Sequential([
        Dense(units=10000, activation='tanh', input_shape=(V0_train.shape[1],)),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V0_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V0_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    mse = model.evaluate(V0_test, P_test, verbose=0)
    mse_scores.append(mse)
    print(f"Mean Squared Error on Test Set (Run {run + 1}): {mse}")

# Calculate and print the average MSE over the runs
average_mse = np.mean(mse_scores)
print(f"\nAverage Mean Squared Error over {num_runs} runs: {average_mse}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")


# In[221]:


# Define the number of runs
num_runs = 10
mse_scores = []

# Copy the data for train-test split outside the loop
V1_train_copy = V1_train.copy()
P_train_copy = P_train.copy()

# Train-test split
V1_train, V1_val, P_train, P_val = train_test_split(V1_train_copy, P_train_copy, test_size=0.1, random_state=42)

for run in range(num_runs):
    # Model architecture
    model = Sequential([
        Dense(units=100, activation='tanh', input_shape=(V1_train.shape[1],)),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V1_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V1_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    mse = model.evaluate(V1_test, P_test, verbose=0)
    mse_scores.append(mse)
    print(f"Mean Squared Error on Test Set (Run {run + 1}): {mse}")

# Calculate and print the average MSE over the runs
average_mse = np.mean(mse_scores)
print(f"\nAverage Mean Squared Error over {num_runs} runs: {average_mse}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")


# In[310]:


P_train=np.array(training_data_labels) # Making probability vector.   
P_train.shape


# In[308]:


V2_train=np.row_stack(f_2_train_arrays) 
V2_train.shape


# In[ ]:





# In[311]:


# Define the number of runs
num_runs = 10
mse_scores = []

# Copy the data for train-test split outside the loop
V2_train_copy = V2_train.copy()
P_train_copy = P_train.copy()

# Train-test split
V2_train, V2_val, P_train, P_val = train_test_split(V2_train_copy, P_train_copy, test_size=0.1, random_state=42)

for run in range(num_runs):
    # Model architecture
    model = Sequential([
        Dense(units=100, activation='tanh', input_shape=(V2_train.shape[1],)),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V2_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V2_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    mse = model.evaluate(V2_test, P_test, verbose=0)
    mse_scores.append(mse)
    print(f"Mean Squared Error on Test Set (Run {run + 1}): {mse}")

# Calculate and print the average MSE over the runs
average_mse = np.mean(mse_scores)
print(f"\nAverage Mean Squared Error over {num_runs} runs: {average_mse}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")


# In[223]:


P_train=np.array(training_data_labels) # Making probability vector.   
P_train.shape
V3_train=np.row_stack(f_3_train_arrays) 
V3_train.shape


# In[224]:


# Define the number of runs
num_runs = 10
mse_scores = []

# Copy the data for train-test split outside the loop
V3_train_copy = V3_train.copy()
P_train_copy = P_train.copy()

# Train-test split
V3_train, V3_val, P_train, P_val = train_test_split(V3_train_copy, P_train_copy, test_size=0.1, random_state=42)

for run in range(num_runs):
    # Model architecture
    model = Sequential([
        Dense(units=100, activation='tanh', input_shape=(V3_train.shape[1],)),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V3_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V3_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    mse = model.evaluate(V3_test, P_test, verbose=0)
    mse_scores.append(mse)
    print(f"Mean Squared Error on Test Set (Run {run + 1}): {mse}")

# Calculate and print the average MSE over the runs
average_mse = np.mean(mse_scores)
print(f"\nAverage Mean Squared Error over {num_runs} runs: {1-average_mse}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")


# In[225]:


P_train=np.array(training_data_labels) # Making probability vector.   
P_train.shape
V4_train=np.row_stack(f_4_train_arrays) 
V4_train.shape


# In[226]:


# Define the number of runs
num_runs = 10
mse_scores = []

# Copy the data for train-test split outside the loop
V4_train_copy = V4_train.copy()
P_train_copy = P_train.copy()

# Train-test split
V4_train, V4_val, P_train, P_val = train_test_split(V4_train_copy, P_train_copy, test_size=0.1, random_state=42)

for run in range(num_runs):
    # Model architecture
    model = Sequential([
        Dense(units=100, activation='tanh', input_shape=(V4_train.shape[1],)),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V4_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V4_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    mse = model.evaluate(V4_test, P_test, verbose=0)
    mse_scores.append(mse)
    print(f"Mean Squared Error on Test Set (Run {run + 1}): {mse}")

# Calculate and print the average MSE over the runs
average_mse = np.mean(mse_scores)
print(f"\nAverage Mean Squared Error over {num_runs} runs: {average_mse}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")
# Define the number of runs
num_runs = 10
mse_scores = []

for run in range(num_runs):
    # Copy the data for train-test split
    V4_train_copy = V4_train.copy()
    P_train_copy = P_train.copy()

    # Train-test split
    V4_train, V4_val, P_train, P_val = train_test_split(V4_train_copy, P_train_copy, test_size=0.1, random_state=42)

    # Model architecture
    model = Sequential([
        Dense(units=100, activation='tanh', input_shape=(V4_train.shape[1],)),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V4_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V4_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    mse = model.evaluate(V4_test, P_test, verbose=0)
    mse_scores.append(mse)
    print(f"Mean Squared Error on Test Set (Run {run + 1}): {mse}")

# Calculate and print the average MSE over the runs
average_mse = np.mean(mse_scores)
print(f"\nAverage Mean Squared Error over {num_runs} runs: {average_mse}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")


# In[227]:


P_train=np.array(training_data_labels) # Making probability vector.   
P_train.shape
V5_train=np.row_stack(f_5_train_arrays) 
V5_train.shape


# In[228]:


# Define the number of runs
num_runs = 10
mse_scores = []

# Copy the data for train-test split outside the loop
V5_train_copy = V5_train.copy()
P_train_copy = P_train.copy()

# Train-test split
V5_train, V5_val, P_train, P_val = train_test_split(V5_train_copy, P_train_copy, test_size=0.1, random_state=42)

for run in range(num_runs):
    # Model architecture
    model = Sequential([
        Dense(units=100, activation='tanh', input_shape=(V5_train.shape[1],)),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V5_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V5_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    mse = model.evaluate(V5_test, P_test, verbose=0)
    mse_scores.append(mse)
    print(f"Mean Squared Error on Test Set (Run {run + 1}): {mse}")

# Calculate and print the average MSE over the runs
average_mse = np.mean(mse_scores)
print(f"\nAverage Mean Squared Error over {num_runs} runs: {average_mse}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")


# In[229]:


P_train=np.array(training_data_labels) # Making probability vector.   
P_train.shape
V5_train=np.row_stack(f_5_train_arrays) 
V5_train.shape


# In[ ]:





# In[230]:


# Define the number of runs
num_runs = 10
mse_scores = []

# Copy the data for train-test split outside the loop
V5_train_copy = V5_train.copy()
P_train_copy = P_train.copy()

# Train-test split
V5_train, V5_val, P_train, P_val = train_test_split(V5_train_copy, P_train_copy, test_size=0.1, random_state=42)

for run in range(num_runs):
    # Model architecture
    model = Sequential([
        Dense(units=100, activation='tanh', input_shape=(V5_train.shape[1],)),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V5_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V5_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    mse = model.evaluate(V5_test, P_test, verbose=0)
    mse_scores.append(mse)
    print(f"Mean Squared Error on Test Set (Run {run + 1}): {mse}")

# Calculate and print the average MSE over the runs
average_mse = np.mean(mse_scores)
print(f"\nAverage Mean Squared Error over {num_runs} runs: {average_mse}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")


# In[237]:


V6_train=np.row_stack(f_6_train_arrays) 
V6_train.shape
P_train=np.array(training_data_labels) # Making probability vector.   
P_train.shape


# In[307]:


# Define the number of runs
num_runs = 10
mse_scores = []

for run in range(num_runs):
    # Copy the data for train-test split
    V6_train_copy = V6_train.copy()
    P_train_copy = P_train.copy()

    # Train-test split
    V6_train, V6_val, P_train, P_val = train_test_split(V6_train_copy, P_train_copy, test_size=0.1, random_state=42)

    # Model architecture
    model = Sequential([
        Dense(units=100, activation='tanh', input_shape=(V6_train.shape[1],)),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss=MeanSquaredError())

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V6_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V6_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    mse = model.evaluate(V6_test, P_test, verbose=0)
    mse_scores.append(mse)
    print(f"Mean Squared Error on Test Set (Run {run + 1}): {mse}")

# Calculate and print the average MSE over the runs
average_mse = np.mean(mse_scores)
print(f"\nAverage accuracy over {num_runs} runs: {1-average_mse}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")


# In[241]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

# Define the number of runs
num_runs = 10
bce_scores = []

for run in range(num_runs):
    # Copy the data for train-test split
    V6_train_copy = V6_train.copy()
    P_train_copy = P_train.copy()

    # Train-test split
    V6_train, V6_val, P_train, P_val = train_test_split(V6_train_copy, P_train_copy, test_size=0.1, random_state=42)

    # Model architecture
    model = Sequential([
        Dense(units=100, activation='tanh', input_shape=(V6_train.shape[1],)),
        Dense(units=10, activation='relu'), 
        Dense(units=10, activation='relu'), 
        Dense(units=10, activation='relu'),
        Dense(units=10, activation='relu'),
        Dense(units=10, activation='relu'),
        Dense(units=1, activation='sigmoid')  # Change activation to 'sigmoid' for binary classification
    ])

    model.compile(optimizer=Adam(), loss=BinaryCrossentropy())  # Change loss to BinaryCrossentropy

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(V6_train, P_train, epochs=1000, batch_size=100, verbose=2,
                        validation_data=(V6_val, P_val), callbacks=[early_stopping])

    # Evaluate on the test set
    bce = model.evaluate(V6_test, P_test, verbose=0)
    bce_scores.append(bce)
    print(f"Binary Cross-Entropy on Test Set (Run {run + 1}): {bce}")

# Calculate and print the average Binary Cross-Entropy over the runs
average_bce = np.mean(bce_scores)
print(f"\nAverage Binary Cross-Entropy over {num_runs} runs: {average_bce}")

# Retrieve learned coefficients
B_learned = model.layers[0].get_weights()[0]
print(f"Learned coefficients (B): {B_learned}")


# In[ ]:





# # Training B Coefficient. 

# In[110]:


# Some things from above.  


# In[111]:


# f_0


# In[112]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_0=f_0_train_arrays
targets = training_data_labels

# Convert lists to numpy arrays
V0 = np.array(features_0)
P = np.array(targets)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V0, P)

# Coefficients
coefficients_B0 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V0)

# Assess the model
r_squared = model.score(V0, P)

# Print results
print("Coefficients B_0:", coefficients_B0)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[113]:


# f_1


# In[114]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_1=f_1_train_arrays
targets = training_data_labels

# Convert lists to numpy arrays
V1 = np.array(features_1)
P = np.array(targets)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V1, P)

# Coefficients
coefficients_B1 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V1)

# Assess the model
r_squared = model.score(V1, P)

# Print results
print("Coefficients B_1:", coefficients_B1)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[115]:


V1


# In[116]:


# f_2


# In[117]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_2=f_2_train_arrays
targets = training_data_labels

# Convert lists to numpy arrays
V2 = np.array(features_2)
P = np.array(targets)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V2, P)

# Coefficients
coefficients_B2 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V2)

# Assess the model
r_squared = model.score(V2, P)

# Print results
print("Coefficients B_2:", coefficients_B2)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[118]:


# f_3


# In[119]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_3=f_3_train_arrays
targets = training_data_labels

# Convert lists to numpy arrays
V3 = np.array(features_3)
P = np.array(targets)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V3, P)

# Coefficients
coefficients_B3 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V3)

# Assess the model
r_squared = model.score(V3, P)

# Print results
print("Coefficients B_3:", coefficients_B3)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[120]:


# f_4


# In[121]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_4=f_4_train_arrays
targets = training_data_labels

# Convert lists to numpy arrays
V4 = np.array(features_4)
P = np.array(targets)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V4, P)

# Coefficients
coefficients_B4 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V4)

# Assess the model
r_squared = model.score(V4, P)

# Print results
print("Coefficients B_4:", coefficients_B4)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[122]:


# f_5


# In[123]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_5=f_5_train_arrays
targets = training_data_labels

# Convert lists to numpy arrays
V5 = np.array(features_5)
P = np.array(targets)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V5, P)

# Coefficients
coefficients_B5 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V5)

# Assess the model
r_squared = model.score(V5, P)

# Print results
print("Coefficients B_5:", coefficients_B5)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[124]:


# f_6


# In[125]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_6=f_6_train_arrays
targets = training_data_labels

# Convert lists to numpy arrays
V6 = np.array(features_6)
P = np.array(targets)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V6, P)

# Coefficients
coefficients_B6 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V6)

# Assess the model
r_squared = model.score(V6, P)

# Print results
print("Coefficients B_6:", coefficients_B6)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[126]:


# Coefficients and Matrices.


# In[127]:


B0_vec=coefficients_B0 
B1_vec=coefficients_B1 
B2_vec=coefficients_B2 
B3_vec=coefficients_B3 
B4_vec=coefficients_B4 
B5_vec=coefficients_B5 
B6_vec=coefficients_B6


# # Preliminary Functions for Linear Regression

# In[128]:


def which_region(prediction, VB_test):
    maximum = max(VB_test)
    minimum = min(VB_test)

    for n in range(200):
        diff = maximum - minimum
        lower_bound = minimum + (diff * n) / 200
        upper_bound = minimum + (diff * (n + 1) / 200)

        if lower_bound <= prediction < upper_bound:
            return n

    # If prediction is outside the specified range, return a special value (e.g., -1)
    return -1


# In[129]:


def decide_on_pred(n, VB, data_labels): #Decision function on input. 
    region_indices = np.array([which_region(value, VB) for value in VB]) # This function runs slowly.
    label_1_count = np.sum((training_data_labels == 1) & (region_indices == n))
    label_0_count = np.sum((training_data_labels == 0) & (region_indices == n))

    if label_1_count > label_0_count:
        return 1
    else:
        return 0


# In[130]:


def decide_on_pred(n, VB, data_labels):
    region_indices = np.array([which_region(value, VB) for value in VB])  # This function runs slowly.

    # Compute indices for the specified region 'n' outside the loop
    region_indices_n = region_indices == n

    label_1_count = np.sum((data_labels == 1) & region_indices_n)
    label_0_count = np.sum((data_labels == 0) & region_indices_n)

    if label_1_count > label_0_count:
        return 1
    else:
        return 0


# In[131]:


def decision_lists(n, non_min_list, min_list):
    non_min_count = non_min_list.count(n)
    min_count = min_list.count(n)

    if non_min_count > min_count:
        return 0
    elif non_min_count > min_count:
        return 1
    else:
        return None


# In[132]:


def optimal(first_half, second_half):
    all_values = np.concatenate([first_half, second_half])

    min_error = float('inf')
    best_n = None

    for value in all_values:
        error = np.sum(first_half > value) + np.sum(second_half < value)
        if error < min_error:
            min_error = error
            best_n = value

    return best_n


# In[133]:


def regression_accuracy(VB): 
    non_min_half = VB[:5000] 
    min_half=VB[5000:] 
    optimal_n = optimal(non_min_half, min_half)
    error=[] 
    for pred in non_min_half: 
        if pred > optimal_n: 
            error.append(pred) 
    for pred in min_half: 
        if pred < optimal_n: 
            error.append(pred)  
    print(f"The optimal value of n is: {optimal_n}")
    return 1-len(error)/len(VB)


# # Quantizing

# In[134]:


# f_0


# In[135]:


VB_test_f_0 = np.dot(V0_test, B0_vec) 
regression_accuracy(VB_test_f_0) 


# In[136]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_0)))
max_value = int(np.ceil(max(VB_test_f_0)))

# Splitting the data
first_half = VB_test_f_0[:5000]
second_half = VB_test_f_0[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_1 Predictions')
plt.legend()
plt.show()


# In[ ]:





# In[137]:


# f_1


# In[138]:


VB_test_f_1 = np.dot(V1_test, B1_vec) 
regression_accuracy(VB_test_f_1) 


# In[139]:


print(regression_accuracy(VB_test_f_1))


# In[140]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_1)))
max_value = int(np.ceil(max(VB_test_f_1)))

# Splitting the data
first_half = VB_test_f_1[:5000]
second_half = VB_test_f_1[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_1 Predictions in F_2')
plt.legend()
plt.show()


# In[141]:


# f_2


# In[142]:


VB_test_f_2 = np.dot(V2_test, B2_vec) 
regression_accuracy(VB_test_f_2) 


# In[143]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_2)))
max_value = int(np.ceil(max(VB_test_f_2)))

# Splitting the data
first_half = VB_test_f_2[:5000]
second_half = VB_test_f_2[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_2 Predictions')
plt.legend()
plt.show()


# In[144]:


# f_3 


# In[145]:


VB_test_f_3 = np.dot(V3_test, B3_vec) 
regression_accuracy(VB_test_f_3) 


# In[146]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_3)))
max_value = int(np.ceil(max(VB_test_f_3)))

# Splitting the data
first_half = VB_test_f_3[:5000]
second_half = VB_test_f_3[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_3 Predictions')
plt.legend()
plt.show()


# In[147]:


VB_test_f_4 = np.dot(V4_test, B4_vec) 
regression_accuracy(VB_test_f_4) 


# In[148]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_4)))
max_value = int(np.ceil(max(VB_test_f_4)))

# Splitting the data
first_half = VB_test_f_4[:5000]
second_half = VB_test_f_4[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_4 Predictions')
plt.legend()
plt.show()


# In[ ]:





# In[149]:


VB_test_f_5 = np.dot(V5_test, B5_vec) 
regression_accuracy(VB_test_f_5) 


# In[150]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_5)))
max_value = int(np.ceil(max(VB_test_f_5)))

# Splitting the data
first_half = VB_test_f_5[:5000]
second_half = VB_test_f_5[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_5 Predictions')
plt.legend()
plt.show()


# In[ ]:





# In[243]:


VB_test_f_6 = np.dot(V6_test, B6_vec) 
regression_accuracy(VB_test_f_6) 


# In[244]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_6)))
max_value = int(np.ceil(max(VB_test_f_6)))

# Splitting the data
first_half = VB_test_f_6[:5000]
second_half = VB_test_f_6[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_6 Predictions')
plt.legend()
plt.show()


# # From $F_2$ to $F_3$ 

# Updating $f_0$

# In[153]:


def f_0_3(word): 
    
    reduced = reduce_and_normalise(word) 
    
    count_list = [] 
    
    Xplusminus = [('a', 1), ('a', -1), ('b', 1), ('b', -1),('c',1),('c',-1)] 
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    for item in list_Xplusminus:
        count_list.append((count_occurrences(reduced, item))/red_word_length(word))
    return np.array(count_list)


# In[24]:


word = [('a', 1), ('b', -1), ('a', 1), ('a', -1), ('b', 1), ('b', -1)]
result = f_0(word)
print(result) 

type(f_0([("a",-1)])) 
result=f_0([("a",1),("b",1)]) 
print(result)


# Updating $f_1$ 

# In[154]:


def U_F_3(n):
    if n == 0:
        return [[]]
    elif n == 1:
        return [[('a', 1)], [('a', -1)], [('b', 1)], [('b', -1)], [('c',1)],[('c',-1)]]
    else:
        U_n_minus_1 = U_F_3(n - 1)
        Xplusminus_3 = [('a', 1), ('a', -1), ('b', 1), ('b', -1),('c',1),('c',-1)]
        U_n = []

        for sublist in U_n_minus_1:
            for item in Xplusminus_3:
                last_item = sublist[-1] if sublist else None

                cancels = last_item is not None and (last_item[0] == item[0] and last_item[1] == -item[1])

                if not cancels:
                    new_sublist = sublist + [item]
                    U_n.append(new_sublist)

        return U_n 
    
def W_F_3(n): 
    m = n + 1
    W = []
    for i in range(m): 
        for words in U_F_3(i):
            W.append(words) 
    return W 


def f_1_3(word): 
    
    reduced = reduce_and_normalise(word)
    
    count_list = [] 
    
    for item in U_F_3(2):
        count_list.append((count_occurrences(reduced , item))/red_word_length(word))
        
    return np.array(count_list) 


# In[155]:


# Updating f_2


# In[156]:


# # 3.4 $$ f_2(w) = \frac{1}{|w|} \langle C(w,x_1U_1x_2) : x_1,x_2 \in X^{\pm1} \rangle  $$

# In[28]:


def f_2_3(word):


    reduced_word = reduce_and_normalise(word)
    
    Xplusminus_3 = [("a", 1), ("a", -1), ("b", 1), ("b", -1), ("c",1),("c",-1) ]
    
    list_Xplusminus_3 = [[item] for item in Xplusminus_3] 
    
    U_F_3_result = [
        [[]],
        U_F_3(1),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus_3:
        for x2 in Xplusminus_3:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, U_F_3_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count) 


# In[157]:


# f_3


# In[158]:


def f_3_3(word):


    reduced_word = reduce_and_normalise(word)
    
    Xplusminus_3 = [("a", 1), ("a", -1), ("b", 1), ("b", -1), ("c",1),("c",-1) ]
    
    list_Xplusminus_3 = [[item] for item in Xplusminus_3] 
    
    U_F_3_result = [
        [[]],
        U_F_3(2),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus_3:
        for x2 in Xplusminus_3:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, U_F_3_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count) 


# In[159]:


# f_4


# In[160]:


def f_4_3(word):


    reduced_word = reduce_and_normalise(word)
    
    Xplusminus_3 = [("a", 1), ("a", -1), ("b", 1), ("b", -1), ("c",1), ("c",-1) ]
     
    list_Xplusminus_3 = [[item] for item in Xplusminus_3] 
    
    U_F_3_result = [
        [[]],
        U_F_3(3),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus_3:
        for x2 in Xplusminus_3:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, U_F_3_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count) 


# In[161]:


# f_5


# In[162]:


def f_5_3(word):


    reduced_word = reduce_and_normalise(word)
    
    Xplusminus_3 = [("a", 1), ("a", -1), ("b", 1), ("b", -1), ("c",1), ("c",-1)]
     
    list_Xplusminus_3 = [[item] for item in Xplusminus_3] 
    
    W_F_3_result = [
        [[]],
        W_F_3(1),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus_3:
        for x2 in Xplusminus_3:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, W_F_3_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count) 


# In[163]:


# f_6


# In[164]:


def f_6_3(word):


    reduced_word = reduce_and_normalise(word)
    
    Xplusminus_3 = [("a", 1), ("a", -1), ("b", 1), ("b", -1), ("c",1), ("c",-1)]
     
    list_Xplusminus_3 = [[item] for item in Xplusminus_3] 
    
    W_F_3_result = [
        [[]],
        W_F_3(3),
        [[]]
    ]
    
    combinations = []

    for x1 in Xplusminus_3:
        for x2 in Xplusminus_3:
            combination = [[x1], [x2], []]  # Create a combination list with x1, x2, and an empty list
            combinations.append(combination)

    count = [] 
    
    for V in combinations: 
        possible_lists = generate_norm_reduced_lists(V, W_F_3_result) 
        cf = count_occurrences_in_list(reduced_word, possible_lists) / red_word_length(word) 
        count.append(cf)
        
    return np.array(count)


# In[ ]:





# In[165]:


# F_3 Training Data


# In[166]:


def no_return_cyclic_reduced_3(l):
    if l == 0:
        return []
    
    if l == 1:
        generators = [("a", 1), ("a", -1), ("b", 1), ("b", -1), ("c",1), ("c",-1)]
        choice = random.choice(generators)
        return [choice]
    
    result = no_return_cyclic_reduced_3(l - 1)
    generators = [("a", 1), ("a", -1), ("b", 1), ("b", -1),("c",1),("c",-1)]
    first_choice_inverse = inverse_letter(result[0])
    last_choice_inverse = inverse_letter(result[-1])
    
    generators.remove(first_choice_inverse)
    
    if first_choice_inverse != last_choice_inverse:
        generators.remove(last_choice_inverse)
    else:
        pass

    n = len(generators)
    choice = random.choices(generators, weights=[1/n] * n)[0]
    result.append(choice)
    
    return result

for i in range(1,5):
    print(no_return_cyclic_reduced_3(i))


# In[ ]:





# In[167]:


def list_of_cyclic_reduced_words_3(n, l):
    result = []

    for _ in range(n):
        word = no_return_cyclic_reduced_3(l)
        result.append(word)
        
    return result


# In[ ]:





# Automorphisms

# In[168]:


def nielsen_1_letter_F_3(word): #This needs checking through
        if word == ("a", 1):
            return [("a", 1)]
        elif word == ("a", -1):
            return [("a", -1)]
        elif word == ("b", 1):
            return [("b", 1),("a", 1)]
        elif word == ("b", -1):
            return [("a", -1),("b", -1)] 
        elif word == ("c",1): 
            return [("c",1)] 
        elif word == ("c",-1):  
            return [("c",-1)]
    
def nielsen_1_F_3(word):
        result = []
        for letters in word:
            new_words = nielsen_1_letter_F_3(letters)
            for new_word in new_words:
                result.append(new_word)
        return result     

      
     # Example usage:
word = [("a", 1), ("b", 1), ("c", -1)]
result = nielsen_1_F_3(word)
print(result)


# In[ ]:





# # Fundamental Domains Training Data

# In[ ]:





# In[169]:


# Lexicographical ordering 
 

def letter_to_number(word): 
    if word == ("a",1): 
        return 0 
    elif word == ("a",-1):  
        return 1 
    elif word == ("b",1):  
        return 2 
    elif word == ("b",-1):  
        return 3    


def word_to_list(word): 
    result=[] 
    for letters in word: 
        result.append(letter_to_number(letters)) 
    return result 

def compare_lists(x, y):
    # Find the first index where the lists differ
    for i in range(min(len(x), len(y))):
        if x[i] != y[i]:
            # Return the ordering based on the comparison of elements at the differing index
            return x if x[i] < y[i] else y
    
    # If the lists are identical up to the minimum length, return the shorter list
    return x if len(x) < len(y) else y


def compare_lexicographically(word1,word2): 
    list1=word_to_list(word1) 
    list2=word_to_list(word2)
    if cyc_reduce(word1)==cyc_reduce(word2):
        return cyc_reduce(word1) 
    elif compare_lists(list1,list2)==list1 : 
        return(word1) 
    else: 
        return(word2) 

def compare_lexicographically_list(list_of_words):
    if not list_of_words:
        return None  # Return None for an empty list

    min_word = list_of_words[0]

    for word in list_of_words[1:]:
        min_word = compare_lexicographically(min_word, word)

    return min_word


# In[170]:


# Lexicographical Ordering: 

def letter_to_number(word): 
    if word == ("a",1): 
        return 0 
    elif word == ("a",-1):  
        return 1 
    elif word == ("b",1):  
        return 2 
    elif word == ("b",-1):  
        return 3    


def word_to_list(word): 
    result=[] 
    for letters in word: 
        result.append(letter_to_number(letters)) 
    return result 

def compare_lists(x, y):
    # Find the first index where the lists differ
    for i in range(min(len(x), len(y))):
        if x[i] != y[i]:
            # Return the ordering based on the comparison of elements at the differing index
            return x if x[i] < y[i] else y
    
    # If the lists are identical up to the minimum length, return the shorter list
    return x if len(x) < len(y) else y


def compare_lexicographically(word1,word2): 
    list1=word_to_list(word1) 
    list2=word_to_list(word2)
    if cyc_reduce(word1)==cyc_reduce(word2):
        return cyc_reduce(word1) 
    elif compare_lists(list1,list2)==list1 : 
        return(word1) 
    else: 
        return(word2) 

def compare_lexicographically_list(list_of_words):
    if not list_of_words:
        return None  # Return None for an empty list

    min_word = list_of_words[0]

    for word in list_of_words[1:]:
        min_word = compare_lexicographically(min_word, word)

    return min_word


# In[189]:




def whitehead_set_on_letters(A,x,y): # NEEDS CHECKING
    
# function that inputs a set & element and a letter, outputs set,element acting on letter

    if x == y: 
        return [y] 
    if x==inverse_letter(y): 
        return [y]
    elif y and inverse_letter(y) in A: 
        return [inverse_letter(x),y,x]  # good
    elif y in A and inverse_letter(y) not in A: # 
        return [y, x] 
    elif y not in A and inverse_letter(y) in A: 
        return [inverse_letter(x), y] 
    else:
        return [y]


# In[ ]:





# In[190]:


def whitehead_set_on_words(pair,word):  
    A=pair[0] 
    x=pair[1]
    result = []
    for letter in word:
        new_words = whitehead_set_on_letters(A,x,letter)
        for new_word in new_words:
            result.append(new_word)
    return result


# In[ ]:





# In[191]:


A=[("a",1),("b",1)]


# In[192]:


whitehead_set_on_letters(A,("b",1), ("a",1) )


# In[193]:


Xplusminus_F_3 = [("a",1), ("a",-1), ("b",1), ("b",-1), ("c",1), ("c",-1)]


# In[194]:


from itertools import chain, combinations

def powerset(lst):
    """
    Returns the powerset of the input list.
    """
    return [list(subset) for subset in chain.from_iterable(combinations(lst, r) for r in range(len(lst) + 1))]

Xplusminus_F_3 = [("a", 1), ("a", -1), ("b", 1), ("b", -1), ("c", 1), ("c", -1)]

power_list = powerset(Xplusminus_F_3)



# In[195]:


for subset in reversed(power_list):
    if len(subset) == 0 or len(subset) == len(Xplusminus_F_3):
        power_list.remove(subset)

len(power_list)


# In[245]:


whitehead_aut_sets=[]
for lists in power_list: 
    for letter in Xplusminus_F_3: 
        if letter in lists and inverse_letter(letter) not in lists: 
            pair=(lists,letter) 
            whitehead_aut_sets.append(pair)


# In[246]:


len(whitehead_aut_sets)


# In[247]:


for pairs in whitehead_aut_sets: 
    print(pairs)


# In[248]:


A=whitehead_aut_sets[67]
print(A)


# In[249]:


whitehead_set_on_words(A,[("a",1),("b",1),("c",1)])


# In[291]:


unique_whitehead_pairs = []
resultants = []

for pair in whitehead_aut_sets:
    results = [whitehead_set_on_words(pair, word) for word in [[("a", 1)], [("b", 1)], [("c", 1)]]]
    if results not in resultants:
        resultants.append(results)
        unique_whitehead_pairs.append(pair)


# In[292]:


len(unique_whitehead_pairs)


# In[293]:


def phi_n(n,word): 
    return whitehead_set_on_words(unique_whitehead_pairs[n],word)


# In[294]:


phi_n(1,[("a",1)])


# In[ ]:





# In[ ]:





# In[300]:


# Define phi_n function
def phi_n(n, word):
    return reduce_and_normalise(whitehead_set_on_words(unique_whitehead_pairs[n], word))

# Create a list of lambda functions
n_values = range(len(unique_whitehead_pairs))
type_2_whitehead_transforms_F_3 = [lambda word, n=n: phi_n(n, word) for n in n_values]


# In[301]:


result = phi_n(2,[("a",1),("b",1),("c",1)])


# In[302]:


print(result)


# In[305]:


phi_n(3,[("a",1),("b",1),("",1)])


# In[209]:


# Creating the training data


# In[ ]:





# In[306]:


F_3_training_data_random = [] # generating non-minimal random cyclically reduced words

for length in range(1, 500):
    words = list_of_cyclic_reduced_words_3(8, length)
    F_3_training_data_random.extend(words) 


# In[261]:


len(F_3_training_data_random)


# In[262]:


3992/2


# In[272]:


def words_getting_smaller_3(word):
    original_length = red_word_length(word)

    for func in type_2_whitehead_transforms_F_3:
        transformed_word = cyc_reduce(func(word))
        transformed_length = cyc_reduce_length(transformed_word)

        if transformed_length < original_length:
            return reduce_and_normalise(transformed_word)

    # If no function reduces the word length, return the original word
    return word #These functions need to be tested 


# In[273]:


def words_getting_bigger_3(word):
    original_length = red_word_length(word)
    
    function_list = list(type_2_whitehead_transforms_F_3)
    random.shuffle(function_list)
    
    for func in function_list:
        transformed_word = cyc_reduce(func(word))
        transformed_length = cyc_reduce_length(transformed_word)

        if transformed_length > original_length:
            return reduce_and_normalise(transformed_word)

    # If no function reduces the word length, return the original word
    return word


# In[274]:


words_getting_bigger_3([("a",1)])


# In[275]:


def find_minimal_rep_3(word):
    original_length = red_word_length(word)
    
    for i in range(original_length):
        transformed_word = words_getting_smaller_3(word)
        transformed_length = red_word_length(transformed_word)
        
        if transformed_length == original_length:
            return transformed_word
        
        word = transformed_word
    
    return word


# In[276]:


F_3_training_data_random = [] # generating non-minimal random cyclically reduced words ##check lables on complexity one data

for length in range(1, 1001):
    words = list_of_cyclic_reduced_words_3(10, length)
    F_3_training_data_random.extend(words) 

minimal_training_data_3=[] # using whitehead algorithm to make minimum. 

for words in F_3_training_data_random: 
    minimal_words=find_minimal_rep_3(words)  
    minimal_training_data_3.append(minimal_words)



len(minimal_training_data_3)

# Now I want to split the data_3 in two.  


random.shuffle(minimal_training_data_3)

first_half_minimal_training_data_3=(minimal_training_data_3[:5000])
second_half_minimal_training_data_3=(minimal_training_data_3[5000:])

len(first_half_minimal_training_data_3)

len(second_half_minimal_training_data_3)

complexity_one_data_3=[] 

for words in first_half_minimal_training_data_3: 
    complexity_one=words_getting_bigger_3(words) 
    complexity_one_data_3.append(complexity_one)

len(complexity_one_data_3)

complexity_one_training_data_3_labelled=[] 

for words in complexity_one_data_3: 
    labelled=(words,0) 
    complexity_one_training_data_3_labelled.append(labelled)

minimal_training_data_3_labelled=[] 

for words in second_half_minimal_training_data_3: 
        label=(words,1) 
        minimal_training_data_3_labelled.append(label)

training_data_3=complexity_one_training_data_3_labelled+minimal_training_data_3_labelled


# In[277]:


# test data


# In[278]:


F_3_test_data_random = [] # generating non-minimal random cyclically reduced words

for length in range(1, 1001):
    words = list_of_cyclic_reduced_words_3(10, length)
    F_3_test_data_random.extend(words) 

minimal_test_data_3=[] # using whitehead algorithm to make minimum. 

for words in F_3_test_data_random: 
    minimal_words=find_minimal_rep_3(words)  
    minimal_test_data_3.append(minimal_words)



len(minimal_test_data_3)

# Now I want to split the data_3 in two.  


random.shuffle(minimal_test_data_3)

first_half_minimal_test_data_3=(minimal_test_data_3[:5000])
second_half_minimal_test_data_3=(minimal_test_data_3[5000:])

len(first_half_minimal_test_data_3)

len(second_half_minimal_test_data_3)

complexity_one_test_data_3=[] 

for words in first_half_minimal_test_data_3: 
    complexity_one=words_getting_bigger_3(words) 
    complexity_one_test_data_3.append(complexity_one)

len(complexity_one_data_3)

complexity_one_test_data_3_labelled=[] 

for words in complexity_one_test_data_3: 
    labelled=(words,0) 
    complexity_one_test_data_3_labelled.append(labelled)

minimal_test_data_3_labelled=[] 

for words in second_half_minimal_test_data_3: 
        label=(words,1) 
        minimal_test_data_3_labelled.append(label)

test_data_3=complexity_one_test_data_3_labelled+minimal_test_data_3_labelled


# In[ ]:





# In[ ]:





# In[279]:


training_data_3_words=[data[0] for data in training_data_3] 
training_data_3_labels=[data[1] for data in training_data_3]


# In[280]:


test_data_3_words=[data[0] for data in test_data_3] 
test_data_3_labels=[data[1] for data in test_data_3]


# In[ ]:





# In[281]:


f_0_3_train_arrays=[f_0_3(t) for t in training_data_3_words]
V0_3_train=np.row_stack(f_0_3_train_arrays)

f_0_3_train_arrays_non_min=f_0_3_train_arrays[:5000] # Min and non-min arrays
f_0_3_train_arrays_min=f_0_3_train_arrays[5000:]

V0_3_train=np.row_stack(f_0_3_train_arrays)  

V0_3_non_min_train=np.row_stack(f_0_train_arrays_non_min) 
V0_3_min_train=np.row_stack(f_0_train_arrays_min) 


# In[282]:


f_1_3_train_arrays=[f_1_3(t) for t in training_data_3_words]
V1_3_train=np.row_stack(f_1_3_train_arrays)

f_1_3_train_arrays_non_min=f_1_3_train_arrays[:5000] # Min and non-min arrays
f_1_3_train_arrays_min=f_1_3_train_arrays[5000:]

V1_3_train=np.row_stack(f_1_3_train_arrays)  

V1_3_non_min_train=np.row_stack(f_1_train_arrays_non_min) 
V1_3_min_train=np.row_stack(f_1_train_arrays_min) 


# In[283]:


f_2_3_train_arrays=[f_2_3(t) for t in training_data_3_words]
V2_3_train=np.row_stack(f_2_3_train_arrays)

f_2_3_train_arrays_non_min=f_2_3_train_arrays[:5000] # Min and non-min arrays
f_2_3_train_arrays_min=f_2_3_train_arrays[5000:]

V2_3_train=np.row_stack(f_2_3_train_arrays)  

V2_3_non_min_train=np.row_stack(f_2_train_arrays_non_min) 
V2_3_min_train=np.row_stack(f_2_train_arrays_min) 


# In[284]:


f_3_3_train_arrays=[f_3_3(t) for t in training_data_3_words]
V3_3_train=np.row_stack(f_3_3_train_arrays)

f_3_3_train_arrays_non_min=f_3_3_train_arrays[:5000] # Min and non-min arrays
f_3_3_train_arrays_min=f_3_3_train_arrays[5000:]

V3_3_train=np.row_stack(f_3_3_train_arrays)  

V3_3_non_min_train=np.row_stack(f_3_train_arrays_non_min) 

V3_3_min_train=np.row_stack(f_3_train_arrays_min) 


# In[ ]:


f_4_3_train_arrays=[f_4_3(t) for t in training_data_3_words]
V4_3_train=np.row_stack(f_4_3_train_arrays)

f_4_3_train_arrays_non_min=f_4_3_train_arrays[:5000] # Min and non-min arrays
f_4_3_train_arrays_min=f_4_3_train_arrays[5000:]

V4_3_train=np.row_stack(f_4_3_train_arrays)  

V4_3_non_min_train=np.row_stack(f_4_train_arrays_non_min) 
V4_3_min_train=np.row_stack(f_4_train_arrays_min) 


# In[ ]:


f_5_3_train_arrays=[f_5_3(t) for t in training_data_3_words]
V5_3_train=np.row_stack(f_5_3_train_arrays)

f_5_3_train_arrays_non_min=f_5_3_train_arrays[:5000] # Min and non-min arrays
f_5_3_train_arrays_min=f_5_3_train_arrays[5000:]

V5_3_train=np.row_stack(f_5_3_train_arrays)  

V5_3_non_min_train=np.row_stack(f_5_train_arrays_non_min) 
V5_3_min_train=np.row_stack(f_5_train_arrays_min) 


# In[ ]:


f_6_3_train_arrays=[f_6_3(t) for t in training_data_3_words]
V6_3_train=np.row_stack(f_6_3_train_arrays)

f_6_3_train_arrays_non_min=f_6_3_train_arrays[:5000] # Min and non-min arrays
f_6_3_train_arrays_min=f_6_3_train_arrays[5000:]

V6_3_train=np.row_stack(f_6_3_train_arrays)  

V6_3_non_min_train=np.row_stack(f_6_train_arrays_non_min) 
V6_3_min_train=np.row_stack(f_6_train_arrays_min) 


# In[ ]:





# In[ ]:





# In[ ]:


f_0_3_test_arrays=[f_0_3(t) for t in test_data_3_words]
V0_3_test=np.row_stack(f_0_3_test_arrays)

f_0_3_test_arrays_non_min=f_0_3_test_arrays[:5000] # Min and non-min arrays
f_0_3_test_arrays_min=f_0_3_test_arrays[5000:]

V0_3_test=np.row_stack(f_0_3_test_arrays)  

V0_3_non_min_test=np.row_stack(f_0_test_arrays_non_min) 
V0_3_min_test=np.row_stack(f_0_test_arrays_min) 


# In[ ]:


f_1_3_test_arrays=[f_1_3(t) for t in test_data_3_words]
V1_3_test=np.row_stack(f_1_3_test_arrays)

f_1_3_test_arrays_non_min=f_1_3_test_arrays[:5000] # Min and non-min arrays
f_1_3_test_arrays_min=f_1_3_test_arrays[5000:]

V1_3_test=np.row_stack(f_1_3_test_arrays)  

V1_3_non_min_test=np.row_stack(f_1_test_arrays_non_min) 
V1_3_min_test=np.row_stack(f_1_test_arrays_min) 


# In[ ]:


f_2_3_test_arrays=[f_2_3(t) for t in test_data_3_words]
V2_3_test=np.row_stack(f_2_3_test_arrays)

f_2_3_test_arrays_non_min=f_2_3_test_arrays[:5000] # Min and non-min arrays
f_2_3_test_arrays_min=f_2_3_test_arrays[5000:]

V2_3_test=np.row_stack(f_2_3_test_arrays)  

V2_3_non_min_test=np.row_stack(f_2_test_arrays_non_min) 
V2_3_min_test=np.row_stack(f_2_test_arrays_min) 


# In[ ]:


f_3_3_test_arrays=[f_3_3(t) for t in test_data_3_words]
V3_3_test=np.row_stack(f_3_3_test_arrays)

f_3_3_test_arrays_non_min=f_3_3_test_arrays[:5000] # Min and non-min arrays
f_3_3_test_arrays_min=f_3_3_test_arrays[5000:]

V3_3_test=np.row_stack(f_3_3_test_arrays)  

V3_3_non_min_test=np.row_stack(f_3_test_arrays_non_min) 
V3_3_min_test=np.row_stack(f_3_test_arrays_min) 


# In[ ]:


f_4_3_test_arrays=[f_4_3(t) for t in test_data_3_words]
V4_3_test=np.row_stack(f_4_3_test_arrays)

f_4_3_test_arrays_non_min=f_4_3_test_arrays[:5000] # Min and non-min arrays
f_4_3_test_arrays_min=f_4_3_test_arrays[5000:]

V4_3_test=np.row_stack(f_4_3_test_arrays)  

V4_3_non_min_test=np.row_stack(f_4_test_arrays_non_min) 
V4_3_min_test=np.row_stack(f_4_test_arrays_min) 


# In[ ]:


f_5_3_test_arrays=[f_5_3(t) for t in test_data_3_words]
V5_3_test=np.row_stack(f_5_3_test_arrays)

f_5_3_test_arrays_non_min=f_5_3_test_arrays[:5000] # Min and non-min arrays
f_5_3_test_arrays_min=f_5_3_test_arrays[5000:]

V5_3_test=np.row_stack(f_5_3_test_arrays)  

V5_3_non_min_test=np.row_stack(f_5_test_arrays_non_min) 
V5_3_min_test=np.row_stack(f_5_test_arrays_min) 


# In[ ]:


f_6_3_test_arrays=[f_6_3(t) for t in test_data_3_words]
V6_3_test=np.row_stack(f_6_3_test_arrays)

f_6_3_test_arrays_non_min=f_6_3_test_arrays[:5000] # Min and non-min arrays
f_6_3_test_arrays_min=f_6_3_test_arrays[5000:]

V6_3_test=np.row_stack(f_6_3_test_arrays)  

V6_3_non_min_test=np.row_stack(f_6_test_arrays_non_min) 
V6_3_min_test=np.row_stack(f_6_test_arrays_min) 


# In[ ]:





# In[ ]:


P_train_3=np.array(training_data_3_labels) # Making probability vector.   
P_train_3.shape


# In[ ]:


P_test_3=np.array(test_data_3_labels) # Making probability vector.   
P_test_3.shape


# In[ ]:





# In[ ]:


# Training coefficients


# In[386]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_0_3=f_0_3_train_arrays
targets_3 = training_data_labels

# Convert lists to numpy arrays
V0_3 = np.array(features_0_3)
P_3 = np.array(targets_3)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V0_3, P_3)

# Coefficients
coefficients_B0_3 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V0_3)

# Assess the model
r_squared = model.score(V0_3, P_3)

# Print results
print("Coefficients B_0_3:", coefficients_B0_3)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[387]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_1_3=f_1_3_train_arrays
targets_3 = training_data_labels

# Convert lists to numpy arrays
V1_3 = np.array(features_1_3)
P_3 = np.array(targets_3)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V1_3, P_3)

# Coefficients
coefficients_B1_3 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V1_3)

# Assess the model
r_squared = model.score(V1_3, P_3)

# Print results
print("Coefficients B_1_3:", coefficients_B1_3)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[405]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_2_3=f_2_3_train_arrays
targets_3 = training_data_labels

# Convert lists to numpy arrays
V2_3 = np.array(features_2_3)
P_3 = np.array(targets_3)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V2_3, P_3)

# Coefficients
coefficients_B2_3 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V2_3)

# Assess the model
r_squared = model.score(V2_3, P_3)

# Print results
print("Coefficients B_2_3:", coefficients_B2_3)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[417]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_3_3=f_3_3_train_arrays
targets_3 = training_data_labels

# Convert lists to numpy arrays
V3_3 = np.array(features_3_3)
P_3 = np.array(targets_3)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V3_3, P_3)

# Coefficients
coefficients_B3_3 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V3_3)

# Assess the model
r_squared = model.score(V3_3, P_3)

# Print results
print("Coefficients B_3_3:", coefficients_B3_3)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[418]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_4_3=f_4_3_train_arrays
targets_3 = training_data_labels

# Convert lists to numpy arrays
V4_3 = np.array(features_4_3)
P_3 = np.array(targets_3)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V4_3, P_3)

# Coefficients
coefficients_B4_3 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V4_3)

# Assess the model
r_squared = model.score(V4_3, P_3)

# Print results
print("Coefficients B_4_3:", coefficients_B4_3)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[419]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_5_3=f_5_3_train_arrays
targets_3 = training_data_labels

# Convert lists to numpy arrays
V5_3 = np.array(features_5_3)
P_3 = np.array(targets_3)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V5_3, P_3)

# Coefficients
coefficients_B5_3 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V5_3)

# Assess the model
r_squared = model.score(V5_3, P_3)

# Print results
print("Coefficients B_5_3:", coefficients_B5_3)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[420]:


# Extracting features (f(w_i)) and target values (P(w_i))
features_6_3=f_6_3_train_arrays
targets_3 = training_data_labels

# Convert lists to numpy arrays
V6_3 = np.array(features_6_3)
P_3 = np.array(targets_3)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V6_3, P_3)

# Coefficients
coefficients_B6_3 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V6_3)

# Assess the model
r_squared = model.score(V6_3, P_3)

# Print results
print("Coefficients B_6_3:", coefficients_B6_3)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[ ]:





# In[421]:


B0_3_vec=coefficients_B0_3 
B1_3_vec=coefficients_B1_3 
B2_3_vec=coefficients_B2_3
B3_3_vec=coefficients_B3_3 
B4_3_vec=coefficients_B4_3
B5_3_vec=coefficients_B5_3
B6_3_vec=coefficients_B6_3


# In[397]:


VB_test_f_0_3 = np.dot(V0_3_test, B0_3_vec) 
regression_accuracy(VB_test_f_0_3) 


# In[398]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_0_3)))
max_value = int(np.ceil(max(VB_test_f_0_3)))

# Splitting the data
first_half = VB_test_f_0_3[:5000]
second_half = VB_test_f_0_3[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_0 Predictions')
plt.legend()
plt.show()


# In[399]:


VB_test_f_1_3 = np.dot(V1_3_test, B1_3_vec) 
regression_accuracy(VB_test_f_1_3) 


# In[426]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_1_3)))
max_value = int(np.ceil(max(VB_test_f_1_3)))

# Splitting the data
first_half = VB_test_f_1_3[:5000]
second_half = VB_test_f_1_3[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_1 Predictions')
plt.legend()
plt.show()


# In[407]:


VB_test_f_2_3 = np.dot(V2_3_test, B2_3_vec) 
regression_accuracy(VB_test_f_2_3) 


# In[427]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_2_3)))
max_value = int(np.ceil(max(VB_test_f_2_3)))

# Splitting the data
first_half = VB_test_f_2_3[:5000]
second_half = VB_test_f_2_3[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_2 Predictions')
plt.legend()
plt.show()


# In[422]:


VB_test_f_3_3 = np.dot(V3_3_test, B3_3_vec) 
regression_accuracy(VB_test_f_3_3) 


# In[428]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_3_3)))
max_value = int(np.ceil(max(VB_test_f_3_3)))

# Splitting the data
first_half = VB_test_f_3_3[:5000]
second_half = VB_test_f_3_3[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 100  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_3 Predictions')
plt.legend()
plt.show()


# In[ ]:





# In[423]:


VB_test_f_4_3 = np.dot(V4_3_test, B4_3_vec) 
regression_accuracy(VB_test_f_4_3) 


# In[433]:


# Convert float values to integers for binning
min_value = int(np.floor(min(VB_test_f_4_3)))
max_value = int(np.ceil(max(VB_test_f_4_3)))

# Splitting the data
first_half = VB_test_f_4_3[:5000]
second_half = VB_test_f_4_3[5000:]

# Increase the number of bins for a more detailed graph
num_bins = 200  # Adjust this value based on your preference

# Plotting the histogram for the first half
plt.hist(first_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, label='Non-Minimal Elements')

# Plotting the histogram for the second half in red
plt.hist(second_half, bins=np.linspace(min_value, max_value, num_bins), edgecolor='black', alpha=0.7, color='red', label='Minimal Elements')

plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.title('Histogram of f_1 Predictions')
plt.legend()
plt.show()


# In[424]:


VB_test_f_5_3 = np.dot(V5_3_test, B5_3_vec) 
regression_accuracy(VB_test_f_5_3) 


# In[425]:


VB_test_f_6_3 = np.dot(V6_3_test, B6_3_vec) 
regression_accuracy(VB_test_f_6_3) 


# In[ ]:




