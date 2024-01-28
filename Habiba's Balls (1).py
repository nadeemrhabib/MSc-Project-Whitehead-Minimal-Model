#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np # Importing all the requisite functions
import numpy.random as npr
import tensorflow.keras as keras
import matplotlib.pyplot as plt 
import random
import unittest  
import time   

from itertools import chain, combinations
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


# We need a way to represent elements of free groups in Python. We choose to represent them as lists of 2-tuples, where the first memeber fo the tuple is the generator, and the second a power of either plus or minus 1. This representation can be difficult to read, e.g. $a^{100}$ will be a list with 100 elements, but is a simpler representation for some of our functions. We call this representation the \textbf{normal| representation. Thus a reduced normal form is one where no consecutive letters of a word are inverses of each other. 

# In[ ]:





# # 1. Essential Functions

# In[56]:


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


# In[57]:


def red_word_length(word):
    
    """

    Returns the reduced and normalized word length.
    """
    # Reduce the word to condensed tuple form 
    reduced =  reduce_and_normalise(word)
    
    # Calculate the sum of absolute values of the second part of each tuple
    result = sum(abs(w[1]) for w in reduced) 
    return result


# In[58]:


def inverse_letter(letter):  
    return((letter[0],-letter[1])) 


# In[59]:


# function that returns a cyclically reduced word 

def cyc_reduce(word1):
    word=reduce_and_normalise(word1)
    while len(word) >= 2 and word[0] == inverse_letter(word[-1]):
        # Remove the first and last letters
        word = word[1:-1]

    return word 


# In[60]:


def cyc_reduce_length(word): 
    return len(cyc_reduce(word))


# # 2 Generating Test and Training Data.

# We begin with some general functions that generate no_return walks and cyclically reduced words. 

# In[61]:


def X_gen(n):
    if n==0: 
        return [] 
    if n==1: 
        return [("a",1),("a",-1)]
    X_n_gen = []
    for i in range(1, n + 1):
        letter = chr(ord('a') + i - 1)
        X_n_gen.append((letter, 1))
        X_n_gen.append((letter, -1))

    return X_n_gen


# In[62]:


def no_return_walk(l, n): # This function has been generalised so can input any generators. (NOT TESTED).

    """
    Inputs length of walk and the generators ( must include inverses ). 
    Outputs a no return walk of input length on generators.
    Output is not cyclically reduced.

    """    
    generators=X_gen(n)
    
    def inverse_letter(letter):  
        return((letter[0],-letter[1]))
    
    result = []
    n=len(generators)
    while len(result) < l:
        if len(result) == 0:
            # With probability 1/n, append one of 
            choice = random.choices(generators, weights=[1/n] * n)[0]
            result.append(choice)
        else:
            last_choice = inverse_letter(result[-1])
            # Determine the valid choices based on the last choice
            valid_choices = [i for i in generators if i != last_choice]
            # With probability 1/(n-1), append one of the valid choices
            choice = random.choice(valid_choices)
            result.append(choice)

    return result

no_return_walk(2,3)


# In[63]:


for n in range(1,10): 
    print(no_return_walk(2,n))


# In[ ]:





# In[64]:


def no_return_cyclic_reduced(l, rank):
    
    generators = X_gen(rank)
    
    if l == 0:
        return []
    
    if l == 1:
        choice = random.choice(generators)
        return [choice]
    
    result = no_return_walk(l - 1, rank)
    first_letter_inv = inverse_letter(result[0])
    last_letter_inv = inverse_letter(result[-1])
    
    letterset = [w for w in generators if w != first_letter_inv and w != last_letter_inv]

    k = len(letterset)
    choice = random.choices(letterset, weights=[1/k] * k)[0]
    result.append(choice)
    
    return result


# In[ ]:





# In[65]:


no_return_cyclic_reduced(10,2)


# In[66]:


# Function that inputs integers k,l,n and outputs list of k random cyclically reduced words of length l and rank n. 


# In[67]:


def list_of_cyclic_reduced_words(k, l, n):
    result = []

    for _ in range(k):
        word = no_return_cyclic_reduced(l, n)
        result.append(word)
        
    return result


# In[68]:


y=list_of_cyclic_reduced_words(20,3, 2)


# In[69]:


y[0]


# # Whitehead Automorphisms

# Now, we need to define the set of Whitehead automorphisms. We do this using Lyndon & Schupp's set and letter representation [insert reference].

# In[70]:


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

def whitehead_set_on_words(pair,word):  
    A=pair[0] 
    x=pair[1]
    result = []
    for letter in word:
        new_words = whitehead_set_on_letters(A,x,letter)
        for new_word in new_words:
            result.append(new_word)
    return result


# In[111]:


# I want to write a function that inputs an integer and then outputs all the Whitehead automorphisms of rank the integer.

def whitehead_pairs(n): 
    
    generators = X_gen(n)  # generators
    
    # powerset, we will remove sets from
    setlist = [list(subset) for subset in chain.from_iterable(combinations(generators, r) for r in range(len(generators) + 1))]

    for subset in reversed(setlist): 
        if len(subset) == 0 or len(subset) == len(generators): 
            setlist.remove(subset) 
            
    whitehead_aut_sets = [] 
    
    for lists in setlist: 
        for letter in generators: 
             if letter in lists and inverse_letter(letter) not in lists: 
                pair = (lists, letter) 
                whitehead_aut_sets.append(pair)
                
    X = generators[::2]  # just take the generators to test. 
    
    unique_whitehead_pairs = [] 
    
    resultant = [] 

    for pair in whitehead_aut_sets: 
        result = [reduce_and_normalise(whitehead_set_on_words(pair, [X[i]])) for i in range(len(X))]
        
        if result not in resultant:
            resultant.append(result) 
            unique_whitehead_pairs.append(pair) 
            
    return unique_whitehead_pairs


# In[ ]:





# In[112]:


# Define phi_n function
def phi_n(n,m, word):
    return whitehead_set_on_words(whitehead_pairs(n)[m], word)


# In[113]:


# Create a list of lambda functions

def whitehead_automorphisms(n): 
    n_values = range(len(whitehead_pairs(n))) 
    return [lambda word, m=m: phi_n(n,m, word) for m in n_values]


# In[114]:


whitehead_automorphisms(3)[30]([("a",1),("a",1)])


# In[117]:


for i in range(2,8): 
    print(len(whitehead_automorphisms(i)))


# In[ ]:





# # Minimisation Algorithm

# In[75]:


whitehead_automorphisms(2)


# In[76]:


whitehead_automorphisms(2)[1]([("b",1)])


# In[77]:


def words_getting_smaller(word, rank):
    original_length = red_word_length(word)

    for func in whitehead_automorphisms(rank):
        transformed_word = cyc_reduce(func(word))
        transformed_length = cyc_reduce_length(transformed_word)

        if transformed_length < original_length:
            return reduce_and_normalise(transformed_word)

    # If no function reduces the word length, return the original word
    return word


# In[33]:


words_getting_smaller([("a",1),("b",1)],2)


# In[97]:


import time

# Your words_getting_smaller function definition goes here

# Example usage and timing
start_time = time.time()

result = find_minimal_rep([("a", 1),("b", 1),("a", 1)], 2)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Result: {result}")
print(f"Elapsed Time: {elapsed_time} seconds")


# In[ ]:





# In[107]:


def words_getting_bigger(word, rank):
    original_length = red_word_length(word)
    
    # Get the shuffled whitehead automorphisms
    automorphisms = list(whitehead_automorphisms(rank))
    random.shuffle(automorphisms)

    # Iterate through automorphisms to find the first one increasing word length
    for automorphism in automorphisms:
        # Apply the automorphism to the word
        transformed_word = cyc_reduce(automorphism(word))
        transformed_length = cyc_reduce_length(transformed_word)

        # Check if the length increased
        if transformed_length > original_length:
            # Return the transformed word and the corresponding automorphism
            return reduce_and_normalise(transformed_word)

    # If no transformation increases the word length, raise an error
    raise ValueError("No transformation increases the word length.")


# In[254]:


def words_getting_bigger(word, rank):
    original_length = red_word_length(word)
    original_word = word  # Store the original word
    
    # Get the shuffled whitehead automorphisms
    automorphisms = list(whitehead_automorphisms(rank))
    random.shuffle(automorphisms)

    # Iterate through automorphisms to find the first one increasing word length
    for automorphism in automorphisms:
        # Apply the automorphism to the word
        transformed_word = cyc_reduce(automorphism(word))
        transformed_length = cyc_reduce_length(transformed_word)

        # Check if the length increased
        if transformed_length > original_length:
            # Return the transformed word and the corresponding automorphism
            return reduce_and_normalise(transformed_word)

    # If no transformation increases the word length, raise an error with the original word
    raise ValueError(f"No transformation increases the length of the word: {original_word}")


# In[257]:


word=[("a",1),("b",1)]


# In[260]:


words_getting_bigger(word,2)


# In[87]:


def find_minimal_rep(word,rank):
    
    original_length = red_word_length(word)
    
    for i in range(original_length):
        transformed_word = words_getting_smaller(word,rank)
        transformed_length = red_word_length(transformed_word)
        
        if transformed_length == original_length:
            return transformed_word
        
        word = transformed_word
    
    return word


# In[ ]:





# # Generating data for $F_2$

# In[132]:


from tqdm import tqdm

non_minimal_training_data_2_1 = []  # First we begin with random cyclically reduced words.

# Wrap the range with tqdm to add a progress bar
for length in tqdm(range(1, 1001), desc="Generating non-minimal data"):
    words = list_of_cyclic_reduced_words(10, length, 2)
    non_minimal_training_data_2_1.extend(words)

minimal_training_data_2_1 = []

# Wrap the loop with tqdm to add a progress bar
for words in tqdm(non_minimal_training_data_2_1, desc="Generating minimal data"):
    minimal_words = find_minimal_rep(words, 2)
    minimal_training_data_2_1.append(minimal_words)


# In[131]:


len(training_data_2_1)


# In[108]:


random.shuffle(minimal_training_data_2_1)

# Split into halves
first_half_minimal_training_data_2_1 = minimal_training_data_2_1[:5000]
second_half_minimal_training_data_2_1 = minimal_training_data_2_1[5000:]

# Create complexity_one_data
complexity_one_data_2_1 = [words_getting_bigger(words,2) for words in first_half_minimal_training_data_2_1]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_2_1 = [(words, 0) for words in complexity_one_data_2_1]

# Create minimal_training_data_labelled
minimal_training_data_labelled_2_1 = [(words, 1) for words in second_half_minimal_training_data_2_1]

# Combine into training_data
training_data_2_1 = complexity_one_training_data_labelled_2_1 + minimal_training_data_labelled_2_1

# Print lengths for verification
print(len(first_half_minimal_training_data_2_1))
print(len(second_half_minimal_training_data_2_1))
print(len(complexity_one_data_2_1))
print(len(training_data_2_1))


# In[ ]:





# In[ ]:





# In[109]:


non_minimal_training_data_2_2 = [] # First we begin with random cyclically reduced words.

for length in range(1, 1001):
    words = list_of_cyclic_reduced_words(10, length, 2)
    non_minimal_training_data_2_2.extend(words) 

minimal_training_data_2_2=[] 

for words in non_minimal_training_data_2_2: 
    minimal_words=find_minimal_rep(words,2)  
    minimal_training_data_2_2.append(minimal_words)


# In[110]:


random.shuffle(minimal_training_data_2_2)

# Split into halves
first_half_minimal_training_data_2_2 = minimal_training_data_2_2[:5000]
second_half_minimal_training_data_2_2 = minimal_training_data_2_2[5000:]

# Create complexity_one_data
complexity_one_data_2_2 = [words_getting_bigger(words,2) for words in first_half_minimal_training_data_2_2]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_2_2 = [(words, 0) for words in complexity_one_data_2_2]

# Create minimal_training_data_labelled
minimal_training_data_labelled_2_2 = [(words, 1) for words in second_half_minimal_training_data_2_2]

# Combine into training_data
training_data_2_2 = complexity_one_training_data_labelled_2_2 + minimal_training_data_labelled_2_2

# Print lengths for verification
print(len(first_half_minimal_training_data_2_2))
print(len(second_half_minimal_training_data_2_2))
print(len(complexity_one_data_2_2))
print(len(training_data_2_2))


# In[133]:


non_minimal_training_data_2_3 = [] # First we begin with random cyclically reduced words.

for length in range(1, 1001):
    words = list_of_cyclic_reduced_words(10, length, 2)
    non_minimal_training_data_2_3.extend(words) 

minimal_training_data_2_3=[] 

for words in non_minimal_training_data_2_3: 
    minimal_words=find_minimal_rep(words,2)  
    minimal_training_data_2_3.append(minimal_words)


# In[167]:


random.shuffle(minimal_training_data_2_3)

# Split into halves
first_half_minimal_training_data_2_3 = minimal_training_data_2_3[:5000]
second_half_minimal_training_data_2_3 = minimal_training_data_2_3[5000:]

# Create complexity_one_data
complexity_one_data_2_3 = [words_getting_bigger(words,2) for words in first_half_minimal_training_data_2_3]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_2_3 = [(words, 0) for words in complexity_one_data_2_3]

# Create minimal_training_data_labelled
minimal_training_data_labelled_2_3 = [(words, 1) for words in second_half_minimal_training_data_2_3]

# Combine into training_data
training_data_2_3 = complexity_one_training_data_labelled_2_3 + minimal_training_data_labelled_2_3

# Print lengths for verification
print(len(first_half_minimal_training_data_2_3))
print(len(second_half_minimal_training_data_2_3))
print(len(complexity_one_data_2_3))
print(len(training_data_2_3))


# In[134]:


non_minimal_training_data_2_4 = [] # First we begin with random cyclically reduced words.

for length in range(1, 1001):
    words = list_of_cyclic_reduced_words(10, length, 2)
    non_minimal_training_data_2_4.extend(words) 

minimal_training_data_2_4=[] 

for words in non_minimal_training_data_2_4: 
    minimal_words=find_minimal_rep(words,2)  
    minimal_training_data_2_4.append(minimal_words)


# In[137]:


random.shuffle(minimal_training_data_2_4)

# Split into halves
first_half_minimal_training_data_2_4 = minimal_training_data_2_4[:5000]
second_half_minimal_training_data_2_4 = minimal_training_data_2_4[5000:]

# Create complexity_one_data
complexity_one_data_2_4 = [words_getting_bigger(words,2) for words in first_half_minimal_training_data_2_4]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_2_4 = [(words, 0) for words in complexity_one_data_2_4]

# Create minimal_training_data_labelled
minimal_training_data_labelled_2_4 = [(words, 1) for words in second_half_minimal_training_data_2_4]

# Combine into training_data
training_data_2_4 = complexity_one_training_data_labelled_2_4 + minimal_training_data_labelled_2_4

# Print lengths for verification
print(len(first_half_minimal_training_data_2_4))
print(len(second_half_minimal_training_data_2_4))
print(len(complexity_one_data_2_4))
print(len(training_data_2_4))


# In[239]:


non_minimal_training_data_2_5 = [] # First we begin with random cyclically reduced words.

# Wrap the range with tqdm to add a progress bar
for length in tqdm(range(1, 1001), desc="Generating non-minimal data"):
    words = list_of_cyclic_reduced_words(10, length, 2)
    non_minimal_training_data_2_5.extend(words)

minimal_training_data_2_5=[] 

# Wrap the loop with tqdm to add a progress bar
for words in tqdm(non_minimal_training_data_2_5, desc="Generating minimal data"):
    minimal_words = find_minimal_rep(words, 2)
    minimal_training_data_2_5.append(minimal_words)


# In[262]:


len(minimal_training_data_2_5)


# In[265]:


minimal_training_data_2_5.remove([])


# In[268]:


minimal_training_data_2_5.append([("a",1)])


# In[269]:


len(minimal_training_data_2_5)


# In[270]:


random.shuffle(minimal_training_data_2_5)

# Split into halves
first_half_minimal_training_data_2_5 = minimal_training_data_2_5[:5000]
second_half_minimal_training_data_2_5 = minimal_training_data_2_5[5000:]

# Create complexity_one_data
complexity_one_data_2_5 = [words_getting_bigger(words,2) for words in first_half_minimal_training_data_2_5]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_2_5 = [(words, 0) for words in complexity_one_data_2_5]

# Create minimal_training_data_labelled
minimal_training_data_labelled_2_5 = [(words, 1) for words in second_half_minimal_training_data_2_5]

# Combine into training_data
training_data_2_5 = complexity_one_training_data_labelled_2_5 + minimal_training_data_labelled_2_5

# Print lengths for verification
print(len(first_half_minimal_training_data_2_5))
print(len(second_half_minimal_training_data_2_5))
print(len(complexity_one_data_2_5))
print(len(training_data_2_5))


# In[244]:


minimal_training_data_2_5


# In[272]:


random.shuffle(minimal_training_data_2_5)

# Split into halves
first_half_minimal_training_data_2_5 = minimal_training_data_2_5[:5000]
second_half_minimal_training_data_2_5 = minimal_training_data_2_5[5000:]

# Create complexity_one_data
complexity_one_data_2_5 = [words_getting_bigger(words,2) for words in first_half_minimal_training_data_2_5]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_2_5 = [(words, 0) for words in complexity_one_data_2_5]

# Create minimal_training_data_labelled
minimal_training_data_labelled_2_5 = [(words, 1) for words in second_half_minimal_training_data_2_5]

# Combine into training_data
training_data_2_5 = complexity_one_training_data_labelled_2_5 + minimal_training_data_labelled_2_5

# Print lengths for verification
print(len(first_half_minimal_training_data_2_5))
print(len(second_half_minimal_training_data_2_5))
print(len(complexity_one_data_2_5))
print(len(training_data_2_5))


# Training Data Summary

# In[329]:


training_data_2_1_words=[t[0] for t in training_data_2_1]
training_data_2_1_labels=[t[1] for t in training_data_2_1]


# In[330]:


training_data_2_2_words=[t[0] for t in training_data_2_2]
training_data_2_2_labels=[t[1] for t in training_data_2_2]


# In[331]:


training_data_2_3_words=[t[0] for t in training_data_2_3]
training_data_2_3_labels=[t[1] for t in training_data_2_3]


# In[332]:


training_data_2_4_words=[t[0] for t in training_data_2_4]
training_data_2_4_labels=[t[1] for t in training_data_2_4]


# In[333]:


training_data_2_5_words=[t[0] for t in training_data_2_5]
training_data_2_5_labels=[t[1] for t in training_data_2_5]


# # Generating data for $F_3$

# In[122]:


from tqdm import tqdm

non_minimal_training_data_3_1 = []  # First we begin with random cyclically reduced words.

# Wrap the range with tqdm to add a progress bar
for length in tqdm(range(1, 501), desc="Generating non-minimal data"):
    words = list_of_cyclic_reduced_words(10, length, 3)
    non_minimal_training_data_3_1.extend(words)

minimal_training_data_3_1 = []

# Wrap the loop with tqdm to add a progress bar
for words in tqdm(non_minimal_training_data_3_1, desc="Generating minimal data"):
    minimal_words = find_minimal_rep(words, 3)
    minimal_training_data_3_1.append(minimal_words)


# In[121]:





# In[235]:


random.shuffle(minimal_training_data_3_1)

# Split into halves
first_half_minimal_training_data_3_1 = minimal_training_data_3_1[:2500]
second_half_minimal_training_data_3_1 = minimal_training_data_3_1[2500:]

# Create complexity_one_data
complexity_one_data_3_1 = [words_getting_bigger(words,3) for words in first_half_minimal_training_data_3_1]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_3_1 = [(words, 0) for words in complexity_one_data_3_1]

# Create minimal_training_data_labelled
minimal_training_data_labelled_3_1 = [(words, 1) for words in second_half_minimal_training_data_3_1]

# Combine into training_data
training_data_3_1 = complexity_one_training_data_labelled_3_1 + minimal_training_data_labelled_3_1

# Print lengths for verification
print(len(first_half_minimal_training_data_3_1))
print(len(second_half_minimal_training_data_3_1))
print(len(complexity_one_data_3_1))
print(len(training_data_3_1))


# In[125]:


non_minimal_training_data_3_2 = []  # First we begin with random cyclically reduced words.

# Wrap the range with tqdm to add a progress bar
for length in tqdm(range(1, 501), desc="Generating non-minimal data"):
    words = list_of_cyclic_reduced_words(10, length, 3)
    non_minimal_training_data_3_2.extend(words)

minimal_training_data_3_2 = []

# Wrap the loop with tqdm to add a progress bar
for words in tqdm(non_minimal_training_data_3_2, desc="Generating minimal data"):
    minimal_words = find_minimal_rep(words, 3)
    minimal_training_data_3_2.append(minimal_words)


# In[126]:


random.shuffle(minimal_training_data_3_2)

# Split into halves
first_half_minimal_training_data_3_2 = minimal_training_data_3_2[:2500]
second_half_minimal_training_data_3_2 = minimal_training_data_3_2[2500:]

# Create complexity_one_data
complexity_one_data_3_2 = [words_getting_bigger(words,3) for words in first_half_minimal_training_data_3_2]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_3_2 = [(words, 0) for words in complexity_one_data_3_2]

# Create minimal_training_data_labelled
minimal_training_data_labelled_3_2 = [(words, 1) for words in second_half_minimal_training_data_3_2]

# Combine into training_data
training_data_3_2 = complexity_one_training_data_labelled_3_2 + minimal_training_data_labelled_3_2

# Print lengths for verification
print(len(first_half_minimal_training_data_3_2))
print(len(second_half_minimal_training_data_3_2))
print(len(complexity_one_data_3_2))
print(len(training_data_3_2))


# In[127]:


non_minimal_training_data_3_3 = []  # First we begin with random cyclically reduced words.

# Wrap the range with tqdm to add a progress bar
for length in tqdm(range(1, 501), desc="Generating non-minimal data"):
    words = list_of_cyclic_reduced_words(10, length, 3)
    non_minimal_training_data_3_3.extend(words)

minimal_training_data_3_3 = []

# Wrap the loop with tqdm to add a progress bar
for words in tqdm(non_minimal_training_data_3_3, desc="Generating minimal data"):
    minimal_words = find_minimal_rep(words, 3)
    minimal_training_data_3_3.append(minimal_words)


# In[286]:


random.shuffle(minimal_training_data_3_3)

# Split into halves
first_half_minimal_training_data_3_3 = minimal_training_data_3_3[:2500]
second_half_minimal_training_data_3_3 = minimal_training_data_3_3[2500:]

# Create complexity_one_data
complexity_one_data_3_3 = [words_getting_bigger(words,3) for words in first_half_minimal_training_data_3_3]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_3_3 = [(words, 0) for words in complexity_one_data_3_3]

# Create minimal_training_data_labelled
minimal_training_data_labelled_3_3 = [(words, 1) for words in second_half_minimal_training_data_3_3]

# Combine into training_data
training_data_3_3 = complexity_one_training_data_labelled_3_3 + minimal_training_data_labelled_3_3

# Print lengths for verification
print(len(first_half_minimal_training_data_3_3))
print(len(second_half_minimal_training_data_3_3))
print(len(complexity_one_data_3_3))
print(len(training_data_3_3))


# In[ ]:





# In[237]:


non_minimal_training_data_3_4 = []  # First we begin with random cyclically reduced words.

# Wrap the range with tqdm to add a progress bar
for length in tqdm(range(1, 501), desc="Generating non-minimal data"):
    words = list_of_cyclic_reduced_words(10, length, 3)
    non_minimal_training_data_3_4.extend(words)

minimal_training_data_3_4 = []

# Wrap the loop with tqdm to add a progress bar
for words in tqdm(non_minimal_training_data_3_4, desc="Generating minimal data"):
    minimal_words = find_minimal_rep(words, 3)
    minimal_training_data_3_4.append(minimal_words)


# In[287]:


random.shuffle(minimal_training_data_3_4)

# Split into halves
first_half_minimal_training_data_3_4 = minimal_training_data_3_4[:2500]
second_half_minimal_training_data_3_4 = minimal_training_data_3_4[2500:]

# Create complexity_one_data
complexity_one_data_3_4 = [words_getting_bigger(words,3) for words in first_half_minimal_training_data_3_4]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_3_4 = [(words, 0) for words in complexity_one_data_3_4]

# Create minimal_training_data_labelled
minimal_training_data_labelled_3_4 = [(words, 1) for words in second_half_minimal_training_data_3_4]

# Combine into training_data
training_data_3_4 = complexity_one_training_data_labelled_3_4 + minimal_training_data_labelled_3_4

# Print lengths for verification
print(len(first_half_minimal_training_data_3_4))
print(len(second_half_minimal_training_data_3_4))
print(len(complexity_one_data_3_4))
print(len(training_data_3_4))


# In[ ]:





# In[238]:


non_minimal_training_data_3_5 = []  # First we begin with random cyclically reduced words.

# Wrap the range with tqdm to add a progress bar
for length in tqdm(range(1, 501), desc="Generating non-minimal data"):
    words = list_of_cyclic_reduced_words(10, length, 3)
    non_minimal_training_data_3_5.extend(words)

minimal_training_data_3_5 = []

# Wrap the loop with tqdm to add a progress bar
for words in tqdm(non_minimal_training_data_3_5, desc="Generating minimal data"):
    minimal_words = find_minimal_rep(words, 3)
    minimal_training_data_3_5.append(minimal_words)


# In[288]:


random.shuffle(minimal_training_data_3_5)

# Split into halves
first_half_minimal_training_data_3_5 = minimal_training_data_3_5[:2500]
second_half_minimal_training_data_3_5 = minimal_training_data_3_5[2500:]

# Create complexity_one_data
complexity_one_data_3_5 = [words_getting_bigger(words,3) for words in first_half_minimal_training_data_3_5]

# Create complexity_one_training_data_labelled
complexity_one_training_data_labelled_3_5 = [(words, 0) for words in complexity_one_data_3_5]

# Create minimal_training_data_labelled
minimal_training_data_labelled_3_5 = [(words, 1) for words in second_half_minimal_training_data_3_5]

# Combine into training_data
training_data_3_5 = complexity_one_training_data_labelled_3_5 + minimal_training_data_labelled_3_5

# Print lengths for verification
print(len(first_half_minimal_training_data_3_5))
print(len(second_half_minimal_training_data_3_5))
print(len(complexity_one_data_3_5))
print(len(training_data_3_5))


# Training Data Summary

# In[335]:


training_data_3_1_words=[t[0] for t in training_data_3_1]
training_data_3_1_labels=[t[1] for t in training_data_3_1]


# In[336]:


training_data_3_2_words=[t[0] for t in training_data_3_2]
training_data_3_2_labels=[t[1] for t in training_data_3_2]


# In[337]:


training_data_3_3_words=[t[0] for t in training_data_3_3]
training_data_3_3_labels=[t[1] for t in training_data_3_3]


# In[338]:


training_data_3_4_words=[t[0] for t in training_data_3_4]
training_data_3_4_labels=[t[1] for t in training_data_3_4]


# In[334]:


training_data_3_5_words=[t[0] for t in training_data_3_5]
training_data_3_5_labels=[t[1] for t in training_data_3_5]


# # Feature Vectors

# In[142]:


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


# In[143]:


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


# In[144]:


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


# # Useful functions for feature vectors

# In[145]:


def U(n,m): # function that ouputs all reduced words of length n, of rank m. 
    
    generators=X_gen(m)
    
    if n == 0:
        return [[]]
    elif n == 1:
        return [[x] for x in generators]
    else:
        U_n_minus_1 = U(n - 1,m)
        U_n = []

        for sublist in U_n_minus_1:
            for item in generators:
                last_item = sublist[-1] if sublist else None

                cancels = last_item is not None and (last_item[0] == item[0] and last_item[1] == -item[1])

                if not cancels:
                    new_sublist = sublist + [item]
                    U_n.append(new_sublist)

        return U_n 


# In[146]:


def W(n,m): 
    k = n + 1
    W = []
    for i in range(k): 
        for words in U(i,m):
            W.append(words) 
    return W 


# In[147]:


U(1,2)


# In[148]:


W(1,3)


# In[149]:


# Defining feature vectors


# In[150]:


def f_0(word,rank): 
    
    reduced = reduce_and_normalise(word) 
    
    count_list = [] 
    
    Xplusminus = X_gen(rank) 
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    for item in list_Xplusminus:
        count_list.append((count_occurrences(reduced, item))/red_word_length(word))
    return np.array(count_list) 


# In[151]:


f_0([("a",1),("b",1),("a",-1)],3)


# In[152]:


def f_1(word,rank): 
    
    reduced = reduce_and_normalise(word)
    
    count_list = [] 
    
    for item in U(2,rank):
        count_list.append((count_occurrences(reduced , item))/red_word_length(word))
        
    return np.array(count_list) 


# In[153]:


f_1([("a",1),("b",1),("a",-1)],3)


# In[154]:


def f_2(word,rank):
    
    reduced_word = reduce_and_normalise(word)
    
    Xplusminus = X_gen(rank)
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        U(1,rank),
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


# In[155]:


f_2([("a",1),("b",1),("a",-1)],3)


# In[156]:


def f_3(word, rank): #Almost identical code to f_2.
    

    reduced_word = reduce_and_normalise(word)
    
    Xplusminus=X_gen(rank)
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        U(2,rank),
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


# In[157]:


f_3([("a",1),("b",1),("a",-1)],3)


# In[158]:


def f_4(word, rank): #Almost identical code to f_2.
    

    reduced_word = reduce_and_normalise(word)
    
    Xplusminus=X_gen(rank)
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        U(3,rank),
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


# In[159]:


f_4([("a",1),("b",1),("a",-1)],3)


# In[160]:


def f_5(word, rank): #Almost identical code to f_2.
    

    reduced_word = reduce_and_normalise(word)
    
    Xplusminus=X_gen(rank)
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        W(1,rank),
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


# In[161]:


f_5([("a",1),("b",1),("a",-1)],3)


# In[162]:


def f_6(word, rank): #Almost identical code to f_2.
    

    reduced_word = reduce_and_normalise(word)
    
    Xplusminus=X_gen(rank)
    
    list_Xplusminus = [[item] for item in Xplusminus] 
    
    U_result = [
        [[]],
        W(3,rank),
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


# # $F_2$ Matrices

# $f_0$

# In[163]:


f_0_train_arrays_2_1=[f_0(t[0],2) for t in training_data_2_1 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_2_1=f_0_train_arrays_2_1[:5000] # Min and non-min arrays
f_0_train_arrays_min_2_1=f_0_train_arrays_2_1[5000:]

V0_train_2_1=np.row_stack(f_0_train_arrays_2_1)  

V0_non_min_train_2_1=np.row_stack(f_0_train_arrays_non_min_2_1) 
V0_min_train_2_1=np.row_stack(f_0_train_arrays_min_2_1) 


# In[165]:


f_0_train_arrays_2_2=[f_0(t[0],2) for t in training_data_2_2 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_2_2=f_0_train_arrays_2_2[:5000] # Min and non-min arrays
f_0_train_arrays_min_2_2=f_0_train_arrays_2_2[5000:]

V0_train_2_2=np.row_stack(f_0_train_arrays_2_2)  

V0_non_min_train_2_2=np.row_stack(f_0_train_arrays_non_min_2_2) 
V0_min_train_2_2=np.row_stack(f_0_train_arrays_min_2_2) 


# In[168]:


f_0_train_arrays_2_3=[f_0(t[0],2) for t in training_data_2_3 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_2_3=f_0_train_arrays_2_3[:5000] # Min and non-min arrays
f_0_train_arrays_min_2_3=f_0_train_arrays_2_3[5000:]

V0_train_2_3=np.row_stack(f_0_train_arrays_2_3)  

V0_non_min_train_2_3=np.row_stack(f_0_train_arrays_non_min_2_3) 
V0_min_train_2_3=np.row_stack(f_0_train_arrays_min_2_3) 


# In[242]:


f_0_train_arrays_2_4=[f_0(t[0],2) for t in training_data_2_4 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_2_4=f_0_train_arrays_2_4[:5000] # Min and non-min arrays
f_0_train_arrays_min_2_4=f_0_train_arrays_2_4[5000:]

V0_train_2_4=np.row_stack(f_0_train_arrays_2_4)  

V0_non_min_train_2_4=np.row_stack(f_0_train_arrays_non_min_2_4) 
V0_min_train_2_4=np.row_stack(f_0_train_arrays_min_2_4) 


# In[273]:


f_0_train_arrays_2_5=[f_0(t[0],2) for t in training_data_2_5 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_2_5=f_0_train_arrays_2_5[:5000] # Min and non-min arrays
f_0_train_arrays_min_2_5=f_0_train_arrays_2_5[5000:]

V0_train_2_5=np.row_stack(f_0_train_arrays_2_5)  

V0_non_min_train_2_5=np.row_stack(f_0_train_arrays_non_min_2_5) 
V0_min_train_2_5=np.row_stack(f_0_train_arrays_min_2_5) 


# $f_1$

# In[172]:


f_1_train_arrays_2_1=[f_1(t[0],2) for t in training_data_2_1 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_2_1=f_1_train_arrays_2_1[:5000] # Min and non-min arrays
f_1_train_arrays_min_2_1=f_1_train_arrays_2_1[5000:]

V1_train_2_1=np.row_stack(f_1_train_arrays_2_1)  

V1_non_min_train_2_1=np.row_stack(f_1_train_arrays_non_min_2_1) 
V1_min_train_2_1=np.row_stack(f_1_train_arrays_min_2_1) 


# In[173]:


f_1_train_arrays_2_2=[f_1(t[0],2) for t in training_data_2_2 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_2_2=f_1_train_arrays_2_2[:5000] # Min and non-min arrays
f_1_train_arrays_min_2_2=f_1_train_arrays_2_2[5000:]

V1_train_2_2=np.row_stack(f_1_train_arrays_2_2)  

V1_non_min_train_2_2=np.row_stack(f_1_train_arrays_non_min_2_2) 
V1_min_train_2_2=np.row_stack(f_1_train_arrays_min_2_2) 


# In[174]:


f_1_train_arrays_2_3=[f_1(t[0],2) for t in training_data_2_3 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_2_3=f_1_train_arrays_2_3[:5000] # Min and non-min arrays
f_1_train_arrays_min_2_3=f_1_train_arrays_2_3[5000:]

V1_train_2_3=np.row_stack(f_1_train_arrays_2_3)  

V1_non_min_train_2_3=np.row_stack(f_1_train_arrays_non_min_2_3) 
V1_min_train_2_3=np.row_stack(f_1_train_arrays_min_2_3) 


# In[175]:


f_1_train_arrays_2_4=[f_1(t[0],2) for t in training_data_2_4 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_2_4=f_1_train_arrays_2_4[:5000] # Min and non-min arrays
f_1_train_arrays_min_2_4=f_1_train_arrays_2_4[5000:]

V1_train_2_4=np.row_stack(f_1_train_arrays_2_4)  

V1_non_min_train_2_4=np.row_stack(f_1_train_arrays_non_min_2_4) 
V1_min_train_2_4=np.row_stack(f_1_train_arrays_min_2_4) 


# In[274]:


f_1_train_arrays_2_5=[f_1(t[0],2) for t in training_data_2_5 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_2_5=f_1_train_arrays_2_5[:5000] # Min and non-min arrays
f_1_train_arrays_min_2_5=f_1_train_arrays_2_5[5000:]

V1_train_2_5=np.row_stack(f_1_train_arrays_2_5)  

V1_non_min_train_2_5=np.row_stack(f_1_train_arrays_non_min_2_5) 
V1_min_train_2_5=np.row_stack(f_1_train_arrays_min_2_5) 


# $f_2$

# In[176]:


f_2_train_arrays_2_1=[f_2(t[0],2) for t in training_data_2_1 ] # Making the f_2 matrix 

f_2_train_arrays_non_min_2_1=f_2_train_arrays_2_1[:5000] # Min and non-min arrays
f_2_train_arrays_min_2_1=f_2_train_arrays_2_1[5000:]

V2_train_2_1=np.row_stack(f_2_train_arrays_2_1)  

V2_non_min_train_2_1=np.row_stack(f_2_train_arrays_non_min_2_1) 
V2_min_train_2_1=np.row_stack(f_2_train_arrays_min_2_1) 


# In[177]:


f_2_train_arrays_2_2=[f_2(t[0],2) for t in training_data_2_2 ] # Making the f_2 matrix 

f_2_train_arrays_non_min_2_2=f_2_train_arrays_2_2[:5000] # Min and non-min arrays
f_2_train_arrays_min_2_2=f_2_train_arrays_2_2[5000:]

V2_train_2_2=np.row_stack(f_2_train_arrays_2_2)  

V2_non_min_train_2_2=np.row_stack(f_2_train_arrays_non_min_2_2) 
V2_min_train_2_2=np.row_stack(f_2_train_arrays_min_2_2) 


# In[178]:


f_2_train_arrays_2_3=[f_2(t[0],2) for t in training_data_2_3 ] # Making the f_2 matrix 

f_2_train_arrays_non_min_2_3=f_2_train_arrays_2_3[:5000] # Min and non-min arrays
f_2_train_arrays_min_2_3=f_2_train_arrays_2_3[5000:]

V2_train_2_3=np.row_stack(f_2_train_arrays_2_3)  

V2_non_min_train_2_3=np.row_stack(f_2_train_arrays_non_min_2_3) 
V2_min_train_2_3=np.row_stack(f_2_train_arrays_min_2_3) 


# In[179]:


f_2_train_arrays_2_4=[f_2(t[0],2) for t in training_data_2_4 ] # Making the f_2 matrix 

f_2_train_arrays_non_min_2_4=f_2_train_arrays_2_4[:5000] # Min and non-min arrays
f_2_train_arrays_min_2_4=f_2_train_arrays_2_4[5000:]

V2_train_2_4=np.row_stack(f_2_train_arrays_2_4)  

V2_non_min_train_2_4=np.row_stack(f_2_train_arrays_non_min_2_4) 
V2_min_train_2_4=np.row_stack(f_2_train_arrays_min_2_4) 


# In[285]:


f_2_train_arrays_2_5=[f_2(t[0],2) for t in training_data_2_5 ] # Making the f_1 matrix 

f_2_train_arrays_non_min_2_5=f_2_train_arrays_2_5[:5000] # Min and non-min arrays
f_2_train_arrays_min_2_5=f_2_train_arrays_2_5[5000:]

V2_train_2_5=np.row_stack(f_2_train_arrays_2_5)  

V2_non_min_train_2_5=np.row_stack(f_2_train_arrays_non_min_2_5) 
V2_min_train_2_5=np.row_stack(f_2_train_arrays_min_2_5) 


# $f_3$

# In[180]:


f_3_train_arrays_2_1=[f_3(t[0],2) for t in training_data_2_1 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_2_1=f_3_train_arrays_2_1[:5000] # Min and non-min arrays
f_3_train_arrays_min_2_1=f_3_train_arrays_2_1[5000:]

V3_train_2_1=np.row_stack(f_3_train_arrays_2_1)  

V3_non_min_train_2_1=np.row_stack(f_3_train_arrays_non_min_2_1) 
V3_min_train_2_1=np.row_stack(f_3_train_arrays_min_2_1) 


# In[181]:


f_3_train_arrays_2_2=[f_3(t[0],2) for t in training_data_2_2 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_2_2=f_3_train_arrays_2_2[:5000] # Min and non-min arrays
f_3_train_arrays_min_2_2=f_3_train_arrays_2_2[5000:]

V3_train_2_2=np.row_stack(f_3_train_arrays_2_2)  

V3_non_min_train_2_2=np.row_stack(f_3_train_arrays_non_min_2_2) 
V3_min_train_2_2=np.row_stack(f_3_train_arrays_min_2_2) 


# In[182]:


f_3_train_arrays_2_3=[f_3(t[0],2) for t in training_data_2_3 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_2_3=f_3_train_arrays_2_3[:5000] # Min and non-min arrays
f_3_train_arrays_min_2_3=f_3_train_arrays_2_3[5000:]

V3_train_2_3=np.row_stack(f_3_train_arrays_2_3)  

V3_non_min_train_2_3=np.row_stack(f_3_train_arrays_non_min_2_3) 
V3_min_train_2_3=np.row_stack(f_3_train_arrays_min_2_3) 


# In[183]:


f_3_train_arrays_2_4=[f_3(t[0],2) for t in training_data_2_4 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_2_4=f_3_train_arrays_2_4[:5000] # Min and non-min arrays
f_3_train_arrays_min_2_4=f_3_train_arrays_2_4[5000:]

V3_train_2_4=np.row_stack(f_3_train_arrays_2_4)  

V3_non_min_train_2_4=np.row_stack(f_3_train_arrays_non_min_2_4) 
V3_min_train_2_4=np.row_stack(f_3_train_arrays_min_2_4) 


# In[276]:


f_3_train_arrays_2_5=[f_3(t[0],2) for t in training_data_2_5 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_2_5=f_3_train_arrays_2_5[:5000] # Min and non-min arrays
f_3_train_arrays_min_2_5=f_3_train_arrays_2_5[5000:]

V3_train_2_5=np.row_stack(f_3_train_arrays_2_5)  

V3_non_min_train_2_5=np.row_stack(f_3_train_arrays_non_min_2_5) 
V3_min_train_2_5=np.row_stack(f_3_train_arrays_min_2_5) 


# $f_4$

# In[277]:


f_4_train_arrays_2_1=[f_4(t[0],2) for t in training_data_2_1 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_2_1=f_4_train_arrays_2_1[:5000] # Min and non-min arrays
f_4_train_arrays_min_2_1=f_4_train_arrays_2_1[5000:]

V4_train_2_1=np.row_stack(f_4_train_arrays_2_1)  

V4_non_min_train_2_1=np.row_stack(f_4_train_arrays_non_min_2_1) 
V4_min_train_2_1=np.row_stack(f_4_train_arrays_min_2_1) 


# In[278]:


f_4_train_arrays_2_2=[f_4(t[0],2) for t in training_data_2_2 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_2_2=f_4_train_arrays_2_2[:5000] # Min and non-min arrays
f_4_train_arrays_min_2_2=f_4_train_arrays_2_2[5000:]

V4_train_2_2=np.row_stack(f_4_train_arrays_2_2)  

V4_non_min_train_2_2=np.row_stack(f_4_train_arrays_non_min_2_2) 
V4_min_train_2_2=np.row_stack(f_4_train_arrays_min_2_2) 


# In[279]:


f_4_train_arrays_2_3=[f_4(t[0],2) for t in training_data_2_3 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_2_3=f_4_train_arrays_2_3[:5000] # Min and non-min arrays
f_4_train_arrays_min_2_3=f_4_train_arrays_2_3[5000:]

V4_train_2_3=np.row_stack(f_4_train_arrays_2_3)  

V4_non_min_train_2_3=np.row_stack(f_4_train_arrays_non_min_2_3) 
V4_min_train_2_3=np.row_stack(f_4_train_arrays_min_2_3) 


# In[280]:


f_4_train_arrays_2_4=[f_4(t[0],2) for t in training_data_2_4 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_2_4=f_4_train_arrays_2_4[:5000] # Min and non-min arrays
f_4_train_arrays_min_2_4=f_4_train_arrays_2_4[5000:]

V4_train_2_4=np.row_stack(f_4_train_arrays_2_4)  

V4_non_min_train_2_4=np.row_stack(f_4_train_arrays_non_min_2_4) 
V4_min_train_2_4=np.row_stack(f_4_train_arrays_min_2_4) 


# In[281]:


f_4_train_arrays_2_5=[f_4(t[0],2) for t in training_data_2_5 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_2_5=f_4_train_arrays_2_5[:5000] # Min and non-min arrays
f_4_train_arrays_min_2_5=f_4_train_arrays_2_5[5000:]

V4_train_2_5=np.row_stack(f_4_train_arrays_2_5)  

V4_non_min_train_2_5=np.row_stack(f_4_train_arrays_non_min_2_5) 
V4_min_train_2_5=np.row_stack(f_4_train_arrays_min_2_5) 


# $f_5$

# In[189]:


from tqdm import tqdm
import time

# Assuming len(training_data_2_1) is the total number of iterations
total_iterations = len(training_data_2_1)

# Initialize the list
f_5_train_arrays_2_1 = []

# Record the start time
start_time = time.time()

# Initialize tqdm with the total number of iterations
for t in tqdm(training_data_2_1, total=total_iterations, desc="Processing loop"):
    f_5_train_arrays_2_1.append(f_5(t[0], 2))

# Rest of your code
f_5_train_arrays_non_min_2_1 = f_5_train_arrays_2_1[:5000]
f_5_train_arrays_min_2_1 = f_5_train_arrays_2_1[5000:]
V5_train_2_1 = np.row_stack(f_5_train_arrays_2_1)
V5_non_min_train_2_1 = np.row_stack(f_5_train_arrays_non_min_2_1)
V5_min_train_2_1 = np.row_stack(f_5_train_arrays_min_2_1)

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


# In[190]:


f_5_train_arrays_2_2=[f_5(t[0],2) for t in training_data_2_2 ] # Making the f_5 matrix 

f_5_train_arrays_non_min_2_2=f_5_train_arrays_2_2[:5000] # Min and non-min arrays
f_5_train_arrays_min_2_2=f_5_train_arrays_2_2[5000:]

V5_train_2_2=np.row_stack(f_5_train_arrays_2_2)  

V5_non_min_train_2_2=np.row_stack(f_5_train_arrays_non_min_2_2) 
V5_min_train_2_2=np.row_stack(f_5_train_arrays_min_2_2) 


# In[191]:


f_5_train_arrays_2_3=[f_5(t[0],2) for t in training_data_2_3 ] # Making the f_5 matrix 

f_5_train_arrays_non_min_2_3=f_5_train_arrays_2_3[:5000] # Min and non-min arrays
f_5_train_arrays_min_2_3=f_5_train_arrays_2_3[5000:]

V5_train_2_3=np.row_stack(f_5_train_arrays_2_3)  

V5_non_min_train_2_3=np.row_stack(f_5_train_arrays_non_min_2_3) 
V5_min_train_2_3=np.row_stack(f_5_train_arrays_min_2_3) 


# In[192]:


f_5_train_arrays_2_4=[f_5(t[0],2) for t in training_data_2_4 ] # Making the f_5 matrix 

f_5_train_arrays_non_min_2_4=f_5_train_arrays_2_4[:5000] # Min and non-min arrays
f_5_train_arrays_min_2_4=f_5_train_arrays_2_4[5000:]

V5_train_2_4=np.row_stack(f_5_train_arrays_2_4)  

V5_non_min_train_2_4=np.row_stack(f_5_train_arrays_non_min_2_4) 
V5_min_train_2_4=np.row_stack(f_5_train_arrays_min_2_4) 


# In[282]:


f_5_train_arrays_2_5=[f_5(t[0],2) for t in training_data_2_5 ] # Making the f_5 matrix 

f_5_train_arrays_non_min_2_5=f_5_train_arrays_2_5[:5000] # Min and non-min arrays
f_5_train_arrays_min_2_5=f_5_train_arrays_2_5[5000:]

V5_train_2_5=np.row_stack(f_5_train_arrays_2_5)  

V5_non_min_train_2_5=np.row_stack(f_5_train_arrays_non_min_2_5) 
V5_min_train_2_5=np.row_stack(f_5_train_arrays_min_2_5) 


# $f_6$

# In[322]:


f_6_train_arrays_2_2=[f_6(t[0],2) for t in training_data_2_2 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_2_2=f_6_train_arrays_2_2[:5000] # Min and non-min arrays
f_6_train_arrays_min_2_2=f_6_train_arrays_2_2[5000:]

V6_train_2_2=np.row_stack(f_6_train_arrays_2_2)  

V6_non_min_train_2_2=np.row_stack(f_6_train_arrays_non_min_2_2) 
V6_min_train_2_2=np.row_stack(f_6_train_arrays_min_2_2) 


# In[194]:


f_6_train_arrays_2_2=[f_6(t[0],2) for t in training_data_2_2 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_2_2=f_6_train_arrays_2_2[:5000] # Min and non-min arrays
f_6_train_arrays_min_2_2=f_6_train_arrays_2_2[5000:]

V6_train_2_2=np.row_stack(f_6_train_arrays_2_2)  

V6_non_min_train_2_2=np.row_stack(f_6_train_arrays_non_min_2_2) 
V6_min_train_2_2=np.row_stack(f_6_train_arrays_min_2_2) 


# In[195]:


f_6_train_arrays_2_3=[f_6(t[0],2) for t in training_data_2_3 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_2_3=f_6_train_arrays_2_3[:5000] # Min and non-min arrays
f_6_train_arrays_min_2_3=f_6_train_arrays_2_3[5000:]

V6_train_2_3=np.row_stack(f_6_train_arrays_2_3)  

V6_non_min_train_2_3=np.row_stack(f_6_train_arrays_non_min_2_3) 
V6_min_train_2_3=np.row_stack(f_6_train_arrays_min_2_3) 


# In[196]:


f_6_train_arrays_2_4=[f_6(t[0],2) for t in training_data_2_4 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_2_4=f_6_train_arrays_2_4[:5000] # Min and non-min arrays
f_6_train_arrays_min_2_4=f_6_train_arrays_2_4[5000:]

V6_train_2_4=np.row_stack(f_6_train_arrays_2_4)  

V6_non_min_train_2_4=np.row_stack(f_6_train_arrays_non_min_2_4) 
V6_min_train_2_4=np.row_stack(f_6_train_arrays_min_2_4) 


# In[283]:


f_6_train_arrays_2_5=[f_6(t[0],2) for t in training_data_2_5 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_2_5=f_6_train_arrays_2_5[:5000] # Min and non-min arrays
f_6_train_arrays_min_2_5=f_6_train_arrays_2_5[5000:]

V6_train_2_5=np.row_stack(f_6_train_arrays_2_5)  

V6_non_min_train_2_5=np.row_stack(f_6_train_arrays_non_min_2_5) 
V6_min_train_2_5=np.row_stack(f_6_train_arrays_min_2_5) 


# Summary of Training Data

# In[ ]:





# # $F_3$ Matrices

# $f_0$

# In[198]:


f_0_train_arrays_3_1=[f_0(t[0],3) for t in training_data_3_1 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_3_1=f_0_train_arrays_3_1[:2500] # Min and non-min arrays
f_0_train_arrays_min_3_1=f_0_train_arrays_3_1[2500:]

V0_train_3_1=np.row_stack(f_0_train_arrays_3_1)  

V0_non_min_train_3_1=np.row_stack(f_0_train_arrays_non_min_3_1) 
V0_min_train_3_1=np.row_stack(f_0_train_arrays_min_3_1) 


# In[199]:


f_0_train_arrays_3_2=[f_0(t[0],3) for t in training_data_3_2 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_3_2=f_0_train_arrays_3_2[:2500] # Min and non-min arrays
f_0_train_arrays_min_3_2=f_0_train_arrays_3_2[2500:]

V0_train_3_2=np.row_stack(f_0_train_arrays_3_2)  

V0_non_min_train_3_2=np.row_stack(f_0_train_arrays_non_min_3_2) 
V0_min_train_3_2=np.row_stack(f_0_train_arrays_min_3_2) 


# In[289]:


f_0_train_arrays_3_3=[f_0(t[0],3) for t in training_data_3_3 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_3_3=f_0_train_arrays_3_3[:2500] # Min and non-min arrays
f_0_train_arrays_min_3_3=f_0_train_arrays_3_3[2500:]

V0_train_3_3=np.row_stack(f_0_train_arrays_3_3)  

V0_non_min_train_3_3=np.row_stack(f_0_train_arrays_non_min_3_3) 
V0_min_train_3_3=np.row_stack(f_0_train_arrays_min_3_3) 


# In[290]:


f_0_train_arrays_3_4=[f_0(t[0],3) for t in training_data_3_4 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_3_4=f_0_train_arrays_3_4[:2500] # Min and non-min arrays
f_0_train_arrays_min_3_4=f_0_train_arrays_3_4[2500:]

V0_train_3_4=np.row_stack(f_0_train_arrays_3_4)  

V0_non_min_train_3_4=np.row_stack(f_0_train_arrays_non_min_3_4) 
V0_min_train_3_4=np.row_stack(f_0_train_arrays_min_3_4) 


# In[291]:


f_0_train_arrays_3_5=[f_0(t[0],3) for t in training_data_3_5 ] # Making the f_0 matrix 

f_0_train_arrays_non_min_3_5=f_0_train_arrays_3_5[:2500] # Min and non-min arrays
f_0_train_arrays_min_3_5=f_0_train_arrays_3_5[2500:]

V0_train_3_5=np.row_stack(f_0_train_arrays_3_5)  

V0_non_min_train_3_5=np.row_stack(f_0_train_arrays_non_min_3_5) 
V0_min_train_3_5=np.row_stack(f_0_train_arrays_min_3_5) 


# $f_1$

# In[204]:


f_1_train_arrays_3_1=[f_1(t[0],3) for t in training_data_3_1 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_3_1=f_1_train_arrays_3_1[:2500] # Min and non-min arrays
f_1_train_arrays_min_3_1=f_1_train_arrays_3_1[2500:]

V1_train_3_1=np.row_stack(f_1_train_arrays_3_1)  

V1_non_min_train_3_1=np.row_stack(f_1_train_arrays_non_min_3_1) 
V1_min_train_3_1=np.row_stack(f_1_train_arrays_min_3_1) 


# In[205]:


f_1_train_arrays_3_2=[f_1(t[0],3) for t in training_data_3_2 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_3_2=f_1_train_arrays_3_2[:2500] # Min and non-min arrays
f_1_train_arrays_min_3_2=f_1_train_arrays_3_2[2500:]

V1_train_3_2=np.row_stack(f_1_train_arrays_3_2)  

V1_non_min_train_3_2=np.row_stack(f_1_train_arrays_non_min_3_2) 
V1_min_train_3_2=np.row_stack(f_1_train_arrays_min_3_2) 


# In[ ]:


f_1_train_arrays_3_3=[f_1(t[0],3) for t in training_data_3_3 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_3_3=f_1_train_arrays_3_3[:2500] # Min and non-min arrays
f_1_train_arrays_min_3_3=f_1_train_arrays_3_3[2500:]

V1_train_3_3=np.row_stack(f_1_train_arrays_3_3)  

V1_non_min_train_3_3=np.row_stack(f_1_train_arrays_non_min_3_3) 
V1_min_train_3_3=np.row_stack(f_1_train_arrays_min_3_3) 


# In[292]:


f_1_train_arrays_3_4=[f_1(t[0],3) for t in training_data_3_4 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_3_4=f_1_train_arrays_3_4[:2500] # Min and non-min arrays
f_1_train_arrays_min_3_4=f_1_train_arrays_3_4[2500:]

V1_train_3_4=np.row_stack(f_1_train_arrays_3_4)  

V1_non_min_train_3_4=np.row_stack(f_1_train_arrays_non_min_3_4) 
V1_min_train_3_4=np.row_stack(f_1_train_arrays_min_3_4) 


# In[293]:


f_1_train_arrays_3_5=[f_1(t[0],3) for t in training_data_3_5 ] # Making the f_1 matrix 

f_1_train_arrays_non_min_3_5=f_1_train_arrays_3_5[:2500] # Min and non-min arrays
f_1_train_arrays_min_3_5=f_1_train_arrays_3_5[2500:]

V1_train_3_5=np.row_stack(f_1_train_arrays_3_5)  

V1_non_min_train_3_5=np.row_stack(f_1_train_arrays_non_min_3_5) 
V1_min_train_3_5=np.row_stack(f_1_train_arrays_min_3_5) 


# $f_2$

# In[206]:


f_2_train_arrays_3_1=[f_2(t[0],3) for t in training_data_3_1 ] # Making the f_2 matrix 

f_2_train_arrays_non_min_3_1=f_2_train_arrays_3_1[:2500] # Min and non-min arrays
f_2_train_arrays_min_3_1=f_2_train_arrays_3_1[2500:]

V2_train_3_1=np.row_stack(f_2_train_arrays_3_1)  

V2_non_min_train_3_1=np.row_stack(f_2_train_arrays_non_min_3_1) 
V2_min_train_3_1=np.row_stack(f_2_train_arrays_min_3_1) 


# In[207]:


f_2_train_arrays_3_2=[f_2(t[0],3) for t in training_data_3_2 ] # Making the f_2 matrix 

f_2_train_arrays_non_min_3_2=f_2_train_arrays_3_2[:2500] # Min and non-min arrays
f_2_train_arrays_min_3_2=f_2_train_arrays_3_2[2500:]

V2_train_3_2=np.row_stack(f_2_train_arrays_3_2)  

V2_non_min_train_3_2=np.row_stack(f_2_train_arrays_non_min_3_2) 
V2_min_train_3_2=np.row_stack(f_2_train_arrays_min_3_2) 


# In[294]:


f_2_train_arrays_3_3=[f_2(t[0],3) for t in training_data_3_3 ] # Making the f_2 matrix 

f_2_train_arrays_non_min_3_3=f_2_train_arrays_3_3[:2500] # Min and non-min arrays
f_2_train_arrays_min_3_3=f_2_train_arrays_3_3[2500:]

V2_train_3_3=np.row_stack(f_2_train_arrays_3_3)  

V2_non_min_train_3_3=np.row_stack(f_2_train_arrays_non_min_3_3) 
V2_min_train_3_3=np.row_stack(f_2_train_arrays_min_3_3) 


# In[295]:


f_2_train_arrays_3_4=[f_2(t[0],3) for t in training_data_3_4 ] # Making the f_2 matrix 

f_2_train_arrays_non_min_3_4=f_2_train_arrays_3_4[:2500] # Min and non-min arrays
f_2_train_arrays_min_3_4=f_2_train_arrays_3_4[2500:]

V2_train_3_4=np.row_stack(f_2_train_arrays_3_4)  

V2_non_min_train_3_4=np.row_stack(f_2_train_arrays_non_min_3_4) 
V2_min_train_3_4=np.row_stack(f_2_train_arrays_min_3_4) 


# In[296]:


f_2_train_arrays_3_5=[f_2(t[0],3) for t in training_data_3_5 ] # Making the f_2 matrix 

f_2_train_arrays_non_min_3_5=f_2_train_arrays_3_5[:2500] # Min and non-min arrays
f_2_train_arrays_min_3_5=f_2_train_arrays_3_5[2500:]

V2_train_3_5=np.row_stack(f_2_train_arrays_3_5)  

V2_non_min_train_3_5=np.row_stack(f_2_train_arrays_non_min_3_5) 
V2_min_train_3_5=np.row_stack(f_2_train_arrays_min_3_5) 


# $f_3$

# In[284]:


f_3_train_arrays_3_1=[f_3(t[0],3) for t in training_data_3_1 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_3_1=f_3_train_arrays_3_1[:2500] # Min and non-min arrays
f_3_train_arrays_min_3_1=f_3_train_arrays_3_1[2500:]

V3_train_3_1=np.row_stack(f_3_train_arrays_3_1)  

V3_non_min_train_3_1=np.row_stack(f_3_train_arrays_non_min_3_1) 
V3_min_train_3_1=np.row_stack(f_3_train_arrays_min_3_1) 


# In[297]:


f_3_train_arrays_3_2=[f_3(t[0],3) for t in training_data_3_2 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_3_2=f_3_train_arrays_3_2[:2500] # Min and non-min arrays
f_3_train_arrays_min_3_2=f_3_train_arrays_3_2[2500:]

V3_train_3_2=np.row_stack(f_3_train_arrays_3_2)  

V3_non_min_train_3_2=np.row_stack(f_3_train_arrays_non_min_3_2) 
V3_min_train_3_2=np.row_stack(f_3_train_arrays_min_3_2) 


# In[298]:


f_3_train_arrays_3_3=[f_3(t[0],3) for t in training_data_3_3 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_3_3=f_3_train_arrays_3_3[:2500] # Min and non-min arrays
f_3_train_arrays_min_3_3=f_3_train_arrays_3_3[2500:]

V3_train_3_3=np.row_stack(f_3_train_arrays_3_3)  

V3_non_min_train_3_3=np.row_stack(f_3_train_arrays_non_min_3_3) 
V3_min_train_3_3=np.row_stack(f_3_train_arrays_min_3_3) 


# In[299]:


f_3_train_arrays_3_4=[f_3(t[0],3) for t in training_data_3_4 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_3_4=f_3_train_arrays_3_4[:2500] # Min and non-min arrays
f_3_train_arrays_min_3_4=f_3_train_arrays_3_4[2500:]

V3_train_3_4=np.row_stack(f_3_train_arrays_3_4)  

V3_non_min_train_3_4=np.row_stack(f_3_train_arrays_non_min_3_4) 
V3_min_train_3_4=np.row_stack(f_3_train_arrays_min_3_4) 


# In[300]:


f_3_train_arrays_3_5=[f_3(t[0],3) for t in training_data_3_5 ] # Making the f_3 matrix 

f_3_train_arrays_non_min_3_5=f_3_train_arrays_3_5[:2500] # Min and non-min arrays
f_3_train_arrays_min_3_5=f_3_train_arrays_3_5[2500:]

V3_train_3_5=np.row_stack(f_3_train_arrays_3_5)  

V3_non_min_train_3_5=np.row_stack(f_3_train_arrays_non_min_3_5) 
V3_min_train_3_5=np.row_stack(f_3_train_arrays_min_3_5) 


# $f_4$

# In[301]:


f_4_train_arrays_3_1=[f_4(t[0],3) for t in training_data_3_1 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_3_1=f_4_train_arrays_3_1[:2500] # Min and non-min arrays
f_4_train_arrays_min_3_1=f_4_train_arrays_3_1[2500:]

V4_train_3_1=np.row_stack(f_4_train_arrays_3_1)  

V4_non_min_train_3_1=np.row_stack(f_4_train_arrays_non_min_3_1) 
V4_min_train_3_1=np.row_stack(f_4_train_arrays_min_3_1) 


# In[302]:


f_4_train_arrays_3_2=[f_4(t[0],3) for t in training_data_3_2 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_3_2=f_4_train_arrays_3_2[:2500] # Min and non-min arrays
f_4_train_arrays_min_3_2=f_4_train_arrays_3_2[2500:]

V4_train_3_2=np.row_stack(f_4_train_arrays_3_2)  

V4_non_min_train_3_2=np.row_stack(f_4_train_arrays_non_min_3_2) 
V4_min_train_3_2=np.row_stack(f_4_train_arrays_min_3_2) 


# In[303]:


f_4_train_arrays_3_3=[f_4(t[0],3) for t in training_data_3_3 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_3_3=f_4_train_arrays_3_3[:2500] # Min and non-min arrays
f_4_train_arrays_min_3_3=f_4_train_arrays_3_3[2500:]

V4_train_3_3=np.row_stack(f_4_train_arrays_3_3)  

V4_non_min_train_3_3=np.row_stack(f_4_train_arrays_non_min_3_3) 
V4_min_train_3_3=np.row_stack(f_4_train_arrays_min_3_3) 


# In[304]:


f_4_train_arrays_3_4=[f_4(t[0],3) for t in training_data_3_4 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_3_4=f_4_train_arrays_3_4[:2500] # Min and non-min arrays
f_4_train_arrays_min_3_4=f_4_train_arrays_3_4[2500:]

V4_train_3_4=np.row_stack(f_4_train_arrays_3_4)  

V4_non_min_train_3_4=np.row_stack(f_4_train_arrays_non_min_3_4) 
V4_min_train_3_4=np.row_stack(f_4_train_arrays_min_3_4) 


# In[305]:


f_4_train_arrays_3_5=[f_4(t[0],3) for t in training_data_3_5 ] # Making the f_4 matrix 

f_4_train_arrays_non_min_3_5=f_4_train_arrays_3_5[:2500] # Min and non-min arrays
f_4_train_arrays_min_3_5=f_4_train_arrays_3_5[2500:]

V4_train_3_5=np.row_stack(f_4_train_arrays_3_5)  

V4_non_min_train_3_5=np.row_stack(f_4_train_arrays_non_min_3_5) 
V4_min_train_3_5=np.row_stack(f_4_train_arrays_min_3_5) 


# $f_5$

# In[306]:


f_5_train_arrays_3_1=[f_5(t[0],3) for t in training_data_3_1 ] # Making the f_5 matrix 

f_5_train_arrays_non_min_3_1=f_5_train_arrays_3_1[:2500] # Min and non-min arrays
f_5_train_arrays_min_3_1=f_5_train_arrays_3_1[2500:]

V5_train_3_1=np.row_stack(f_5_train_arrays_3_1)  

V5_non_min_train_3_1=np.row_stack(f_5_train_arrays_non_min_3_1) 
V5_min_train_3_1=np.row_stack(f_5_train_arrays_min_3_1) 


# In[307]:


f_5_train_arrays_3_2=[f_5(t[0],3) for t in training_data_3_2 ] # Making the f_5 matrix 

f_5_train_arrays_non_min_3_2=f_5_train_arrays_3_2[:2500] # Min and non-min arrays
f_5_train_arrays_min_3_2=f_5_train_arrays_3_2[2500:]

V5_train_3_2=np.row_stack(f_5_train_arrays_3_2)  

V5_non_min_train_3_2=np.row_stack(f_5_train_arrays_non_min_3_2) 
V5_min_train_3_2=np.row_stack(f_5_train_arrays_min_3_2) 


# In[308]:


f_5_train_arrays_3_3=[f_5(t[0],3) for t in training_data_3_3 ] # Making the f_5 matrix 

f_5_train_arrays_non_min_3_3=f_5_train_arrays_3_3[:2500] # Min and non-min arrays
f_5_train_arrays_min_3_3=f_5_train_arrays_3_3[2500:]

V5_train_3_3=np.row_stack(f_5_train_arrays_3_3)  

V5_non_min_train_3_3=np.row_stack(f_5_train_arrays_non_min_3_3) 
V5_min_train_3_3=np.row_stack(f_5_train_arrays_min_3_3) 


# In[309]:


f_5_train_arrays_3_4=[f_5(t[0],3) for t in training_data_3_4 ] # Making the f_5 matrix 

f_5_train_arrays_non_min_3_4=f_5_train_arrays_3_4[:2500] # Min and non-min arrays
f_5_train_arrays_min_3_4=f_5_train_arrays_3_4[2500:]

V5_train_3_4=np.row_stack(f_5_train_arrays_3_4)  

V5_non_min_train_3_4=np.row_stack(f_5_train_arrays_non_min_3_4) 
V5_min_train_3_4=np.row_stack(f_5_train_arrays_min_3_4) 


# In[310]:


f_5_train_arrays_3_5=[f_5(t[0],3) for t in training_data_3_5 ] # Making the f_5 matrix 

f_5_train_arrays_non_min_3_5=f_5_train_arrays_3_5[:2500] # Min and non-min arrays
f_5_train_arrays_min_3_5=f_5_train_arrays_3_5[2500:]

V5_train_3_5=np.row_stack(f_5_train_arrays_3_5)  

V5_non_min_train_3_5=np.row_stack(f_5_train_arrays_non_min_3_5) 
V5_min_train_3_5=np.row_stack(f_5_train_arrays_min_3_5) 


# $f_6$

# In[311]:


f_6_train_arrays_3_1=[f_6(t[0],3) for t in training_data_3_1 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_3_1=f_6_train_arrays_3_1[:2500] # Min and non-min arrays
f_6_train_arrays_min_3_1=f_6_train_arrays_3_1[2500:]

V6_train_3_1=np.row_stack(f_6_train_arrays_3_1)  

V6_non_min_train_3_1=np.row_stack(f_6_train_arrays_non_min_3_1) 
V6_min_train_3_1=np.row_stack(f_6_train_arrays_min_3_1) 


# In[312]:


f_6_train_arrays_3_2=[f_6(t[0],3) for t in training_data_3_2 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_3_2=f_6_train_arrays_3_2[:2500] # Min and non-min arrays
f_6_train_arrays_min_3_2=f_6_train_arrays_3_2[2500:]

V6_train_3_2=np.row_stack(f_6_train_arrays_3_2)  

V6_non_min_train_3_2=np.row_stack(f_6_train_arrays_non_min_3_2) 
V6_min_train_3_2=np.row_stack(f_6_train_arrays_min_3_2) 


# In[313]:


f_6_train_arrays_3_3=[f_6(t[0],3) for t in training_data_3_3 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_3_3=f_6_train_arrays_3_3[:2500] # Min and non-min arrays
f_6_train_arrays_min_3_3=f_6_train_arrays_3_3[2500:]

V6_train_3_3=np.row_stack(f_6_train_arrays_3_3)  

V6_non_min_train_3_3=np.row_stack(f_6_train_arrays_non_min_3_3) 
V6_min_train_3_3=np.row_stack(f_6_train_arrays_min_3_3) 


# In[314]:


f_6_train_arrays_3_4=[f_6(t[0],3) for t in training_data_3_4 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_3_4=f_6_train_arrays_3_4[:2500] # Min and non-min arrays
f_6_train_arrays_min_3_4=f_6_train_arrays_3_4[2500:]

V6_train_3_4=np.row_stack(f_6_train_arrays_3_4)  

V6_non_min_train_3_4=np.row_stack(f_6_train_arrays_non_min_3_4) 
V6_min_train_3_4=np.row_stack(f_6_train_arrays_min_3_4) 


# In[315]:


f_6_train_arrays_3_5=[f_6(t[0],3) for t in training_data_3_5 ] # Making the f_6 matrix 

f_6_train_arrays_non_min_3_5=f_6_train_arrays_3_5[:2500] # Min and non-min arrays
f_6_train_arrays_min_3_5=f_6_train_arrays_3_5[2500:]

V6_train_3_5=np.row_stack(f_6_train_arrays_3_5)  

V6_non_min_train_3_5=np.row_stack(f_6_train_arrays_non_min_3_5) 
V6_min_train_3_5=np.row_stack(f_6_train_arrays_min_3_5) 


# In[ ]:





# In[ ]:





# # Experiments on $F_2$

# Linear Regression Training Coefficients

# $f_1$

# In[348]:


# Convert lists to numpy arrays
P_2_1 = np.array(training_data_2_1_labels)


# In[349]:


# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V1_train_2_1, P_2_1)

# Coefficients
coefficients_B2_1_1 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V1_train_2_1)

# Assess the model
r_squared = model.score(V1_train_2_1, P_2_1)

# Print results
print("Coefficients B2_1_1:", coefficients_B2_1_1)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[353]:


# Convert lists to numpy arrays
P_2_2 = np.array(training_data_2_2_labels)

# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V1_train_2_2, P_2_2)

# Coefficients
coefficients_B2_1_2 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V1_train_2_2)

# Assess the model
r_squared = model.score(V1_train_2_2, P_2_2)

# Print results
print("Coefficients B2_1_2:", coefficients_B2_1_2)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[352]:


# Using Linear Regression from scikit-learn
model = LinearRegression()
model.fit(V1_train_2_3, P_2_1)

# Coefficients
coefficients_B2_1_3 = model.coef_
intercept = model.intercept_

# Make predictions
predictions = model.predict(V1_train_2_3)

# Assess the model
r_squared = model.score(V1_train_2_3, P_2_1)

# Print results
print("Coefficients B2_1_2:", coefficients_B2_1_2)
print("Intercept:", intercept)
print("R-squared:", r_squared)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




