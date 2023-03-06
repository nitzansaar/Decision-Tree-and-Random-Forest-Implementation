### CS 462 - Assignment 3. Learning and Representing Knowledge.

#### Due Friday, March 24, 11:59pm.

To submit: Please check all code into your GitHub repo, along with a PDF containing the 
answers to the written questions. Also, please provide a submission.py that demonstrates
all of your code.

#### Part 1. Inference. 

NASA has reached out to us for help in designing the logic for the Mars rover.
They would like it to be able to conduct experiments, and only radio back to Earth for help 
in case of an actual emergency. We've decided to build a rule-based system to implement this.

We have the following predicates (with abbreviations) to use:

- a - battery failed
- b - solar panels failed
- c - base station failed
- d - backup battery failed
- e: send emergency signal
- q: backup switch fails
- w: cannot sync base station with rover
- x: battery sensor not responding
- y: battery will not power motor
- z: no current from panels

NASA has given us the following rules to use:
- If the battery sensor is not responding and the battery will not power the motor, 
then the battery failed.
- If the battery sensor is not responding, we are not able to sync the base station with the rover.
- If the battery sensor is not responding and there is no current from the panels, then the solar panels have failed.
- If the battery will not power the motor, and we cannot sync the base station with the rover, then the base station has failed.
- If the backup switch fails, and there's no current from the panels, and the battery fails, then the backup battery fails.
- If the backup switch fails, then the battery will not power the motor.
- If battery failed and solar panels failed and base station failed and backup battery failed, then send the emergency signal.
- If we cannot sync the base station with the rover, we are unable to get current from the panels.

Our rover has observed the following facts:
- The battery sensor is not responding.
- The backup switch has failed.

**(5 points)** 1. Show what the initial KB looks like, using the abbreviations for each rule and fact.

**(5 points)** 2. Use forward chaining to show that we should send an emergency signal. For each step, show the rule and fact(s) being matched, and the resulting fact.
for example: q, q->y, y.

**(5 points)** 3. Our NASA contact is skeptical; he's heard that AI can make mistakes, and does not believe that there is truly an emergency.
Use backward chaining to show that signaling the emergency was the correct decision. For each
step, show the stack and the items to be proved.

4. Our NASA contact is still skeptical - just because we were right this time doesn't mean it could make mistakes, he thinks!
We'll use resolution to verify the correctness of our rules.

**(5 points)** a) Convert the KB to Conjunctive Normal Form.

**(5 points)** b) Add the term !e (do not signal) to the KB, and use resolution to derive a contradiction.
For each step, show the two terms being resolved and the new sentence that results. 
for example: (!w v z) ^ (!x v w) -> (z v !x)

##### Part 2: Intro to scikit-learn.

In this part, you'll get some practice working with scikit-learn, using the template in sklearn_intro.py.
to begin, install scikit-learn. (note - don't install the package named sklearn - this is an older version!)

**(5 points)** 1. The template code shows how to load the iris dataset. Change this to load the breast cancer data instead.

**(5 points)** 2. Now we'll implement our first two learning algorithms: ZeroR and RandR. These 
are very simple algorithms that we'll use as a baseline. ZeroR should always return the most common classification, and RandR should 
select from amongst the classifications according to their frequency.

**(5 points)** 3. The template code shows how to use scikit-learn's DecisionTreeClassifier on the iris data. Change this to work with breast cancer.

**(5 points)** The code then shows how to set up a training and test set. This lets us know that we're able to classify unseen data. But to really measure this, we need to implement cross-validation.
We'll do it once by hand. I've started setting it up for you - you finish it.
The Intro code also shows how to use sklearn to do cross-validation, and how to use the Random Forest. We'll come back to those.

#### Part 3: Decision tree 

This is the largest part of the assignment - we'll implement the basic decision tree algorithm. I've provided two starter datasets (tennis and restaurant) for you, plus some unit tests.
Start with these datasets, and then test on breast cancer once you've got it working.

**(5 points)**  I've implemented entropy for you. Use that to implement remainder.

**(10 points)** Use remainder to implement select_attribute.

**(10 points)** Use select_attribute to implement make_tree.

**(5 points)** Now you are ready to implement classify. 

Measure the performance of your tree on the restaurant and breast cancer data using five-fold cross-validation. (If you want to explore the more challenging datasets in sklearn, the California housing dataset is a good one.) 
Compare it to the performance of the scikit-learn DecisionTree as well as the Random Forest. (note - you should not expect to do as well as the scikit-learn implementation.)

#### Part 4: intro to NLTK 

This exercise will give you some experience working with natural language using the NLTK toolkit. This is mostly to get you started; we'll look at texts more carefully in the next assignment.

**(5 points)** Read chapter 1. Load in nltk.book. Consider text1 ('Moby Dick'),
text2 ('Sense and Sensibility'), text3 ('Book of Genesis'), text7('Wall Street Journal')
Write a loop that displays, for each text, which words are similar to: 'great','king','country','fear','love'.

**(5 points)**  Write a loop that, for each text, generates a 50-token random sequence.

**(5 points)**  Now let's make two Frequency Distributions. You built something very similar in assignment 1. We'll use the movie_reviews corpus.
Construct one Frequency Distribution that counts all of the words in the positive reviews,
and one that counts all of the words in the negative reviews. Print the 10 most common words in each distribution.

**(5 points)**  There's a lot of noise in our data. As in assignment1, update your code to remove stopwords, non-words, and convert everything to lower case.
Does that help to distinguish between positive and negative reviews?

**(5 points)**  Rather than keeping our distributions in two separate data structures, let's use a ConditionalFrequencyDistribution. (see Chapter 3)
Add a loop that iterates through the ConditionalFrequencyDistribution and prints the 10 most common words for each category.


#### Part 5: (686 students only) 

Please read [this article](https://12ft.io/proxy?q=https%3A%2F%2Fwww.theatlantic.com%2Fmagazine%2Farchive%2F2013%2F11%2Fthe-man-who-would-teach-machines-to-think%2F309529%2F) about Douglas Hofstadter, 
which also serves as a nice summary of the history of AI and the debates 
over the value of developing machines that think like humans. 
(As an aside: If you have not read Hofstadter's book [Godel, Escher, Bach](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach), 
I strongly recommend it.)

Prepare a summary or critique of this article that addresses the 
following questions:
- Hofstadter is particularly interested in understanding the way humans think. What sorts of reasoning mechanisms does he study?
- The article includes a quote from our text: “The quest for ‘artificial flight’ succeeded when 
the Wright brothers and others stopped imitating birds and started … learning about aerodynamics,” What does this mean? Why is it relevant to AI?
- What was Candide? Why did it change the way we thought about machine translation?
- The article also contains a quote from the last chapter of AIMA: perhaps AI has become too much like 
the man who tries to get to the moon by climbing a tree: “One can report steady progress, all the way to the top of the tree.” What does this mean? How does it relate to Candide 
and the ways in which big data and machine learning have changed AI?
