

README
Decision Tree Classifier using Iris Dataset
This project demonstrates how to build, train, and test a Decision Tree Classifier using the popular Iris dataset from Scikit-learn.

📌 Requirements
Make sure you have the following Python packages installed:

scikit-learn
numpy
Install using:

pip install scikit-learn numpy
📂 Project Description
The program:

Loads the Iris dataset.

Splits it into training (70%) and testing (30%) sets.

Creates a Decision Tree Classifier model.

Trains the model on the training data.

Predicts the labels for the test data.

Prints the predicted class labels.

📜 Code
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
data = load_iris()
X = data.data          # Features
y = data.target        # Labels

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create the Decision Tree model
model = DecisionTreeClassifier()

# Train the model using training data
model.fit(X_train, y_train)

# Predict output for test data
pred = model.predict(X_test)

# Print the predictions
print(pred)
▶️ How to Run
Run the script using:

python filename.py
✔️ Output
The program will display the predicted labels for the test dataset, something like:

[2 0 1 2 1 ...]
If you want this README saved as a .txt, .md, or docx file, just tell me!
