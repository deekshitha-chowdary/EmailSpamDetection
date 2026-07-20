import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Display first 5 rows
# Keep only the useful columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Display first 5 rows
print("\nCleaned Dataset:")
print(df.head())

# Display new column names
print("\nColumns after cleaning:")
print(df.columns)

# Display shape
print("\nShape after cleaning:")
print(df.shape)
# Count ham and spam messages
print("\nNumber of Ham and Spam messages:")
print(df['label'].value_counts())
# Plot Ham vs Spam messages
import matplotlib.pyplot as plt
df['label'].value_counts().plot(kind='bar')

plt.title("Ham vs Spam Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Feature 1: Message Length
df['message_length'] = df['message'].apply(len)

print("\nMessage Length:")
print(df[['message', 'message_length']].head())
# Feature 2: Word Count
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

print("\nWord Count:")
print(df[['message', 'word_count']].head())
# Feature 3: Digit Count
df['digit_count'] = df['message'].apply(lambda x: sum(c.isdigit() for c in x))

print("\nDigit Count:")
print(df[['message', 'digit_count']].head())
# Feature 4: Punctuation Count
df['punctuation_count'] = df['message'].apply(
    lambda x: sum(c in string.punctuation for c in x)
)

print("\nPunctuation Count:")
print(df[['message', 'punctuation_count']].head())
# Feature 5: Uppercase Count
df['uppercase_count'] = df['message'].apply(
    lambda x: sum(c.isupper() for c in x)
)

print("\nUppercase Count:")
print(df[['message', 'uppercase_count']].head())
# Feature 6: Spam Keyword Presence

spam_keywords = [
    "free", "win", "winner", "claim",
    "prize", "urgent", "offer", "cash"
]

df['keyword_present'] = df['message'].apply(
    lambda x: int(any(word in x.lower() for word in spam_keywords))
)

print("\nKeyword Presence:")
print(df[['message', 'keyword_present']].head())
# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("\nConverted Labels:")
print(df[['label']].head())
# Features (Input)
X = df[
    [
        'message_length',
        'word_count',
        'digit_count',
        'punctuation_count',
        'uppercase_count',
        'keyword_present'
    ]
]

# Target (Output)
y = df['label']

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)
# Train Logistic Regression Model

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")
# Predict on test data

y_pred = lr.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])
# Model Evaluation

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance")
print("-----------------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("\nClassification Report")
print(classification_report(y_test, y_pred))
# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()
# -------------------------------
# Naive Bayes Model
# -------------------------------

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

print("\nNaive Bayes Results")
print("------------------------")

print("Accuracy :", round(accuracy_score(y_test, y_pred_nb),4))
print("Precision:", round(precision_score(y_test, y_pred_nb),4))
print("Recall   :", round(recall_score(y_test, y_pred_nb),4))
print("F1 Score :", round(f1_score(y_test, y_pred_nb),4))

print("\nClassification Report")
print(classification_report(y_test, y_pred_nb))
# -------------------------------
# Decision Tree Model
# -------------------------------

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Results")
print("------------------------")

print("Accuracy :", round(accuracy_score(y_test, y_pred_dt), 4))
print("Precision:", round(precision_score(y_test, y_pred_dt), 4))
print("Recall   :", round(recall_score(y_test, y_pred_dt), 4))
print("F1 Score :", round(f1_score(y_test, y_pred_dt), 4))

print("\nClassification Report")
print(classification_report(y_test, y_pred_dt))
# -------------------------------
# Random Forest Model
# -------------------------------

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results")
print("------------------------")

print("Accuracy :", round(accuracy_score(y_test, y_pred_rf), 4))
print("Precision:", round(precision_score(y_test, y_pred_rf), 4))
print("Recall   :", round(recall_score(y_test, y_pred_rf), 4))
print("F1 Score :", round(f1_score(y_test, y_pred_rf), 4))

print("\nClassification Report")
print(classification_report(y_test, y_pred_rf))