import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the extracted feature dataset
dataset_path = 'breath_features.npy'
data = np.load(dataset_path)

print(f"Dataset loaded with shape: {data.shape}")

# Since our matrix is shape (26, 388), let's flatten or transpose it 
# so machine learning models can process the time frames as samples.
# Transposing to (388, 26) means we have 388 time frames, each with 26 features.
X = data.T  # Shape: (388, 26)

# For testing the skeleton, let's create dummy labels (0 for Normal, 1 for Apnea)
# In real training, you will map your audio chunks to actual labels.
y = np.random.randint(0, 2, size=X.shape[0])

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build and train a simple Logistic Regression model as our baseline skeleton
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluate the model skeleton
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model skeleton trained successfully!")
print(f"Baseline Accuracy on test frames: {accuracy * 100:.2f}%")