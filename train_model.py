import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the real labeled dataset
X = np.load('X_data.npy')
y = np.load('y_labels.npy')

print(f"Loaded X shape: {X.shape}")
print(f"Loaded y shape: {y.shape}")

# 2. Split data (80% for training, 20% for testing/validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Baseline Model
# max_iter is increased to ensure the math solver converges properly with our large dataset
# class_weight='balanced' forces the model to pay more attention to the minority class (Apnea)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# 4. Evaluate the model on the unseen test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Real Data Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Breathing/Snoring (0)', 'Apnea (1)']))