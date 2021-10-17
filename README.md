# Naive-Bayes Spam Classificator

Module is created to build a spam filter using Python and the multinomial Naive Bayes algorithm. Main goal is to code a spam filter from scratch that classifies messages with an accuracy greater than 90%.

## Main files

This project contains 2 modules:

- bayesian_classifier.py -- BayesianClassifier class which is used to classify
  whether the message is spam or ham. It also calculates model score for accuracy.
- main.py -- main file for running the programm.

## How does it work?

After running a program, Spam Classificator provides the user with 2 options:

- to test a model on data base and get model score;
- to test a model on statement the user inputs;

If there are many words in the statement that are not present in the database, then the
program will most likely output ‘needs human classification, probably spam’.
