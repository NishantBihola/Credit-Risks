# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Import joblib for saving the model

# Step 1: Load the dataset
data_path = 'mini_project/german_credit_data.csv'  # Update the path if necessary
df = pd.read_csv(data_path)

# Step 2: Check for missing values and handle them
print("Missing values in dataset:\n", df.isnull().sum())

# Step 3: Feature selection
# Select relevant features (both numeric and nominal)
numeric_features = ['Age', 'Duration', 'Credit_Amount', 'Savings_Account']  # Example numeric features
nominal_features = ['Sex', 'Housing', 'Purpose']  # Example nominal features

# Combine features
features = numeric_features + nominal_features

# Step 4: Preprocess the data
# Encoding nominal features using LabelEncoder
encoder = LabelEncoder()
for col in nominal_features:
    df[col] = encoder.fit_transform(df[col])

# Step 5: Split the data into features (X) and target (y)
X = df[features]
y = df['Risk']  # Assuming 'Risk' is the target column (1 = bad credit, 0 = good credit)

# Step 6: Split the dataset into training, validation, and test sets (80%, 10%, 10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 7: Scale the numeric features for KNN
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_val[numeric_features] = scaler.transform(X_val[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Step 8: Train a KNN classifier with different k values
k_values = [3, 5, 7, 9, 11]
best_k = 3
best_accuracy = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Step 9: Validate and choose the best k
    y_val_pred = knn.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Accuracy for k={k}: {val_accuracy}")
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_k = k

# Step 10: Train the final model with the best k
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)

# Step 11: Evaluate the model on the test set
y_test_pred = knn_final.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {test_accuracy}")

# Step 12: Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Step 13: Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Good Credit', 'Bad Credit'], yticklabels=['Good Credit', 'Bad Credit'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Step 14: Save the trained model using joblib
joblib.dump(knn_final, 'knn_credit_model.pkl')
print("Model saved as knn_credit_model.pkl")
