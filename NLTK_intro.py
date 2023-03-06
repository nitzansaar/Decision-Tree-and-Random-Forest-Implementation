## A quick introduction to NLTK.
## Please refer to the NLTK book for more details.
import nltk
from nltk.book import *

## Read chapter 1. Load in nltk.book. Consider text1 ('Moby Dick'),
# text2 ('Sense and Sensibility'), text3 ('Book of Genesis'), text7('Wall Street Journal')

## For each text, which words are similar to:

texts_to_examine = [text1, text2, text3, text7]
words = ['great','king','country','fear','love']

## your answer goes here.


## For each text, generate a 50-word sequence
## your answer goes here.

## Now let's look at the movie review corpus
from nltk.corpus import movie_reviews

## we can get all the fileids this way:

files = movie_reviews.fileids()

## to get the words in one file, we can do:

wordlist = movie_reviews.words(files[0])

## let's make two frequency distributions. One for positive reviews,
## and one for negative reviews.

pos_reviews = FreqDist()
neg_reviews = FreqDist()

## you populate these with all of the words in movie_reviews. Add the words from
## the positive reviews to pos_reviews, and the negative reviews to neg_reviews



## What are the most common words in each review? Does anything stand out as
## helping to distinguish them?

## What if we do this again, but remove non-words and stopwords and convert to
# lower case?

## Rather than having two separate data structures, let's make a Conditional Frequency
## Distribution

reviews = nltk.ConditionalFreqDist()
reviews['positive'] = pos_reviews
reviews['negative'] = neg_reviews

## add a loop that iterates through the ConditionalFreqDist and prints the
## 10 most common words for each category.




