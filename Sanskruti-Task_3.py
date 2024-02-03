from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables) 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
import zipfile

# Specify the path to the ZIP file
zip_file_path = "C:/Users/Sanskruti_Thakur/Desktop/SANS_INTERNSHIPS/bank.zip"

# Specify the path to the CSV file within the ZIP archive
csv_file_path = "bank.csv"

# Read the CSV file directly from the ZIP archive
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    with zip_ref.open(csv_file_path) as csv_file:
        bank_data = pd.read_csv(csv_file, sep=';')
# Step 2: Preprocess the Data
# Handle missing values, encode categorical variables, and split the dataset
X = bank_data.drop(columns=['y'])  # Features
y = bank_data['y']  # Target variable

# Encode categorical variables (e.g., using one-hot encoding)
X = pd.get_dummies(X)

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Step 5: Train the Classifier
clf.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Fine-tune the Model (Optional)
# You can adjust hyperparameters of the decision tree classifier using grid search or random search.

# Step 8: Make Predictions (Optional)
# Use the trained classifier to make predictions on new data.
# Print the first few rows of the DataFrame
print(bank_data.head())

# Print basic information about the DataFrame
print(bank_data.info())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"


# Step 2: Preprocess the Data
X = bank_data.drop(columns=['y'])  # Features
y = bank_data['y']  # Target variable
X = pd.get_dummies(X)  # One-hot encoding for categorical variables

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Step 5: Train the Classifier
clf.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Fine-tune the Model (Optional)
# You can adjust hyperparameters of the decision tree classifier using grid search or random search.

# Step 8: Make Predictions (Optional)
# Use the trained classifier to make predictions on new data.
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example of predicting a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # New sample to predict
predicted_class = clf.predict(new_sample)
print("Predicted class:", predicted_class)

