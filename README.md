# Sentiment Analysis with Hidden Markov Model (HMM)

Welcome to the Sentiment Analysis with Hidden Markov Model (HMM) project! This project implements an HMM algorithm to perform sentiment analysis, following a structured approach to build and train the model using the Baum-Welch algorithm and the Viterbi algorithm.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Tasks](#tasks)
   - [Task 1: Transition Probability Matrix](#task-1-transition-probability-matrix)
   - [Task 2: Viterbi Algorithm](#task-2-viterbi-algorithm)
   - [Task 3: Formulating HMM](#task-3-formulating-hmm)
   - [Task 4: Baum-Welch Algorithm](#task-4-baum-welch-algorithm)
5. [Conclusion](#conclusion)
6. [Contributors](#contributors)
7. [References](#references)

## Introduction

This project explores the fundamentals of Hidden Markov Models (HMMs) by applying them to the task of sentiment analysis. The primary goal is to implement HMM algorithms to predict sentence likelihood and determine the best probable path for hidden states in a sequence.

## Dataset

The dataset used in this project is the Named Entity Recognition (NER) Dataset from Kaggle. It consists of labeled text data that helps in training and evaluating machine learning models to identify and extract entities from text. For this project, we focus on sentence likelihood prediction using Part-Of-Speech (POS) tagging.

### Dataset Contents
- **Sentence**: Indicates the sentence number.
- **Word**: The word in the sentence.
- **POS**: The POS tag associated with the word.
- **Tag**: Named entity recognition tags.

## Project Structure

The project is organized as follows:
- **MLF_project_HMM.ipynb**: Jupyter notebook containing the code implementation and instructions.
- **data/dataset.csv**: The dataset file used for training and evaluation.

## Tasks

### Task 1: Transition Probability Matrix

We built a transition probability matrix between words by calculating the log likelihood given a sentence. This involved creating arrays to store the count of words following each word and then normalizing these counts to form a probability matrix.

### Task 2: Viterbi Algorithm

We implemented the Viterbi algorithm to find the most likely sequence of hidden states in an HMM. The algorithm was tested using a sample sentence, and the results were compared with the hmmlearn implementation to ensure correctness.

### Task 3: Formulating HMM

In this task, we created a hypothetical scenario to demonstrate the use of an HMM. We defined hidden states and observable states, generated random observations, and used these to train an HMM. The model was then used to predict the most likely sequence of hidden states for given observations.

### Task 4: Baum-Welch Algorithm

We implemented the Baum-Welch algorithm to estimate the HMM parameters. This involved completing functions to compute forward and backward probabilities and iteratively refining the transition and emission matrices. The algorithm was tested with two different examples to validate its performance.


## Conclusion

This project successfully implemented an HMM for sentiment analysis, demonstrating the use of the Viterbi and Baum-Welch algorithms. HMMs provide a powerful framework for modeling sequential data, with applications in various fields such as NLP, speech recognition, and bioinformatics. By understanding and applying HMMs, we can develop more accurate and efficient machine learning models.

## Contributors

We would like to thank the following contributors for their efforts in this project:
- **[Contributor 1]**: 
- **[Contributor 2]**:
- **[Contributor 3]**: 

## References

- [Kaggle NER Dataset](https://www.kaggle.com/datasets/debasisdotcom/name-entity-recognition-ner-dataset)
- [hmmlearn documentation](https://hmmlearn.readthedocs.io/en/latest/)
- Rabiner, L. R. (1989). A tutorial on Hidden Markov Models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
- Wikipedia contributors. (2023). Hidden Markov model. In *Wikipedia, The Free Encyclopedia*. Retrieved from [https://en.wikipedia.org/wiki/Hidden_Markov_model](https://en.wikipedia.org/wiki/Hidden_Markov_model)

Thank you for visiting our project! We hope you find this work informative and useful. If you have any questions or suggestions, please feel free to reach out.
