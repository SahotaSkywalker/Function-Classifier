import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the dataset
df =pd.read_csv('./test.csv', on_bad_lines='skip')

# Split the dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(df["Code"], df["language"], test_size=0.2, random_state=42)

# Convert the text data into numerical features using Tf-Idf
vectorizer = TfidfVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train a SVM classifier on the training data
model = SVC()
model.fit(X_train_features, y_train)

# Predict on the test data and evaluate accuracy
y_pred = model.predict(X_test_features)
accuracy = (y_pred == y_test).mean()
print("Accuracy: ", accuracy)

#test the model
test_function = input("Enter the function u want to test: ")
test_function_features = vectorizer.transform([test_function])
predicted_language = model.predict(test_function_features)[0]
print("Predicted language: ", predicted_language)
