# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset (make sure spam.csv is in same folder)
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Dataset Preview:")
print(df.head())

# Step 3: Convert label to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split Dataset
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Convert text to numerical values
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test_vec)

# Step 8: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Test Custom Message
msg = ["Congratulations! You won a free prize"]

msg_vector = vectorizer.transform(msg)

prediction = model.predict(msg_vector)

if prediction[0] == 1:
    print("\nThis message is SPAM")
else:
    print("\nThis message is NOT SPAM")