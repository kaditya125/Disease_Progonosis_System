
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Training.csv')

# Separate features (X) and target variable (y)
X = data.drop('prognosis', axis=1)
y = data['prognosis']

# Initialize the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Check if the model file exists, if not, train and save the model
model_file_path = 'your_model.pkl'
if not os.path.exists(model_file_path):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_file_path)
else:
    # Load the pre-trained model
    model = joblib.load(model_file_path)

# Define all_symptoms as a global variable
all_symptoms = list(X.columns)

# Function to get user input
def get_user_input():
    symptoms = []
    st.sidebar.header('User Input')
    selected_symptoms = st.sidebar.multiselect('Select Symptoms', all_symptoms)

    # Convert selected symptoms to binary values
    symptoms = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]

    return symptoms

# Function to predict prognosis and provide an explanation
def predict_prognosis(symptoms):
    # Ensure that the input symptoms match the model's expectations
    if len(symptoms) != len(all_symptoms):
        st.error("Invalid number of symptoms. Please provide all symptoms.")
        return None

    # Make a prediction
    prediction = model.predict([symptoms])

    # Plot the decision tree
    st.subheader('Decision Tree Visualization')
    plot_tree(model, feature_names=all_symptoms, class_names=model.classes_, filled=True)
    st.pyplot()

    # Provide an explanation based on the decision tree
    explanation = "The model made the prediction based on the decision tree. " \
                  "Each node in the tree represents a decision based on a specific symptom."
    st.info(explanation)

    return prediction[0]

# Function to visualize symptom frequency
def visualize_symptom_frequency():
    st.subheader('Symptom Frequency Visualization')
    symptom_counts = data.drop('prognosis', axis=1).sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=symptom_counts.values, y=symptom_counts.index, palette='viridis')
    plt.xlabel('Frequency')
    plt.title('Symptom Frequency in the Dataset')
    st.pyplot()

def main():
    st.title('Prognosis Prediction App')

    # Get user input
    symptoms = get_user_input()

    # When the user clicks the 'Predict' button
    if st.button('Predict'):
        # Make a prediction and display the result
        result = predict_prognosis(symptoms)

        if result is not None:
            if result == 'Positive':
                st.success(f'The predicted prognosis is: {result} 😊')
                st.balloons()
            else:
                st.warning(f'The predicted prognosis is: {result} 😟')

    # Visualize symptom frequency
    visualize_symptom_frequency()

if __name__ == '__main__':
    main()