# EmailSpamDetection
Email Spam Detection using Machine Learning with Python and Scikit-learn
# Email Spam Detection

## Problem Statement
Email and SMS spam messages are a common issue that can cause security risks and unwanted communication. The objective of this project is to build a machine learning-based spam detection system that classifies messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques and classification algorithms.

## Dataset
**Dataset Name:** SMS Spam Collection Dataset (UCI Machine Learning Repository)

- Total messages: 5,574
- Classes:
  - Ham (Legitimate messages)
  - Spam (Unwanted messages)
- Target variable: Message category (Spam/Ham)

The dataset contains labeled text messages that are used for training and evaluating the classification models.

## Features
The following features were extracted from the messages:

- Message length
- Word count
- Number of characters
- Number of digits
- Number of punctuation marks
- Presence of spam-related keywords
- TF-IDF (Term Frequency-Inverse Document Frequency) text features

These features help the model identify patterns commonly found in spam messages.

## Models Used
The following machine learning classification models were trained and compared:

1. Logistic Regression
2. Naive Bayes
3. Random Forest Classifier

The models were evaluated and compared to select the best-performing classifier.

## Hyperparameter Tuning
Hyperparameter optimization was performed using **GridSearchCV**.

- Cross-validation technique was used to find the best parameter combination.
- F1-score was used as the primary evaluation metric because the dataset contains class imbalance.

Examples of tuned parameters:
- Logistic Regression: C, penalty, solver
- Random Forest: number of estimators, maximum depth, minimum samples split
- Naive Bayes: alpha value

## Evaluation Metrics
The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

F1-score was given importance because it provides a balance between precision and recall, especially for detecting spam messages.

## Results

Model performance comparison:

| Model | Accuracy | Precision | Recall | F1-score | AUC |
|------|----------|-----------|--------|----------|-----|
| Logistic Regression | [Enter value] | [Enter value] | [Enter value] | [Enter value] | [Enter value] |
| Naive Bayes | [Enter value] | [Enter value] | [Enter value] | [Enter value] | [Enter value] |
| Random Forest | [Enter value] | [Enter value] | [Enter value] | [Enter value] | [Enter value] |

**Best Model:** [Enter best model name]

The selected model achieved the highest F1-score and showed better capability in distinguishing spam and ham messages.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/deekshitha-chowdary/EmailSpamDetection.git