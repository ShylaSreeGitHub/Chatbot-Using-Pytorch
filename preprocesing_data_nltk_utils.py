import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer_type = PorterStemmer()

#Tokenize function splits a string into meaningfulunits.
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
    
#Stem function generates root form of the words.
def stem(word):
    return stemmer_type.stem(word.lower())

#bag of words function converts all words into ones and zeros based on its respective patterns.
#For example, all words = ["hi","bye"]
# result for bye = [0,1]
#result for hi = [1,0]
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag_words = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag_words[index] = 1.0
    return bag_words

