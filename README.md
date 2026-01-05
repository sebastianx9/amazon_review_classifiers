# Amazon Review Classifiers
This is a machine learning project implementing two classifiers for Amazon product reviews: a binary sentiment classifier (positive/negative) and a multi-class helpfulness classifier (helpful/neutral/unhelpful), based on a Bag-of-Words approach.

This repository contains the code of a final coursework project in ipython notbook form.

## Description

### Dataset

- Source: Amazon product reviews across 24 categories.

- Volume: 36,547 reviews in total.

- Preprocessing:
  - Tokenization and lowercasing.
  - Removal of punctuation and extra spaces.
  - Split into training (80%), validation (10%), and test (10%) sets.

Labels:

- Sentiment: Positive (1) or Negative (0).

- Helpfulness: Helpful (0), Neutral (1), or Unhelpful (2). 

### Project Structure
This project explores fundamental Natural Language Processing (NLP) and supervised machine learning concepts by building two classifiers from scratch:

- Classifier 1: Sentiment Classifier - A binary logistic regression model to predict if a review is positive or negative.

- Classifier 2: Helpfulness Classifier - A multi-class logistic regression model (using softmax) to predict if a review is helpful, neutral, or unhelpful.

The core methodology involves a Bag-of-Words (BoW) feature representation, selecting the top 5000 most frequent words, and training models using Stochastic Gradient Descent (SGD) with mini-batch training and L2 regularization.


## Getting Started

### 1. Prerequisites
Ensure you have Python 3.8 or later installed.

### 2. Clone the Repository
```bash
git clone https://github.com/sebastianx9/amazon_review_classifiers.git
```
### 3. Dependencies
```bash
pip install -r requirements.txt
```

### 4.Run the project

Execute the Classifier script to train both models and see results.

This will:

- Load and preprocess the data.

- Train the sentiment classifier and evaluate it on the test set. 

- Train the helpfulness classifier and evaluate it on the test set. 

- Print evaluation metrics and generate learning curves. 

## Usage
For reproducibility purposes, the core training logic for the classifiers has been encapsulated in the sentiment_classifier.py and helpfulness_classifier.py file.

In your Python script or Jupyter Notebook, after preparing your training test matrix and true vlues, use the following statement to import the training functions:

```bash
from sentiment_classifier import train_sentiment_classifier
```
and
```bash
from helpfulness_classifier import train_helpfulness_classifier
```

- These functions uses parameters identical to the original experimental code:
  - M_train, sents_train,seed_value=42, num_features=5000, n_iters=2500, lr=0.1, lambda_l2=0.001, batch_size=128
  - M_train, y_train, num_features=5000, num_classes=3, seed_value=42, n_iters=2500, lr=0.05, lambda_l2=0.001, batch_size=128
- Returns:
    - weights (np.ndarray): The trained weight vector.
    - bias (np.ndarray): The trained bias term.
    - logistic_loss (list): A list recording the average training loss for each epoch.


## Authors and Acknowledgements 
- Author: Haotian Xu
- Instructor: Dr. Colin Bannard
- Course: [LELA60331]
- Institution: University of Manchester

## License
All Rights Reserved
