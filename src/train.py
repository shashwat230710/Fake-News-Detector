import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Load the prepared dataset
print("Loading dataset...")
df = pd.read_csv("data/fake_news.csv")

# Drop rows with missing values in 'text'
df.dropna(subset=['text'], inplace=True)

# Define features (X) and target (y)
X = df['text']
y = df['label']

# Split data into training and testing sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a machine learning pipeline
print("Creating pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Train the model
print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Create the models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the trained model
print("Saving model...")
joblib.dump(pipeline, 'models/baseline.joblib')

print("✅ Model trained and saved to models/baseline.joblib")
