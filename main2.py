# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
############################################################################################################################################################
# Load the dataset
data = pd.read_csv('parkinsons.csv')

# Display the first few rows of the dataset
# print(data.head())

# Get a summary of the dataset
# print(data.info())
############################################################################################################################################################
# Define features (X) and target (y)
X = data.drop(columns=['name', 'status'])
y = data['status']

# Display the features and target
# print("Features (X):")
# print(X.head())
# print("\nTarget (y):")
# print(y.head())
############################################################################################################################################################
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the features and transform them
X_scaled = scaler.fit_transform(X)

# Convert the scaled features back to a DataFrame for better readability
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Display the first few rows of the scaled features
# print("Scaled Features (X):")
# print(X_scaled.head())
############################################################################################################################################################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
# print("Training Set Shape (X_train, y_train):", X_train.shape, y_train.shape)
# print("Testing Set Shape (X_test, y_test):", X_test.shape, y_test.shape)
############################################################################################################################################################
# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Display a message indicating the model has been trained
# print("Random Forest model has been trained.")
############################################################################################################################################################
# Predict the labels for the test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")