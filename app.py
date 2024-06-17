import streamlit as st
import pickle
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


st.title("Welcome to Telecome data prediction")
st.header("Telecome data prediction")
df = pd.read_csv("customer_agg.csv")


# Features and target variable
X = df[['Average TCP Retransmission', 'Average RTT', 'Average Throughput']]
y = df['Satisfaction Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict the satisfaction score
y_pred = reg_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(X)
print(y)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Save the model
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(reg_model, file)


# Load the trained model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

def display_model_score():
    score = model.score(X_test,y_test)
    st.write(f"score of model is : {score }")

def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.subheader("score of model at TEST data")
    display_model_score()
    
    st.header("input feature")
    feature1 = st.number_input("Average TCP Retransmission",value=0.00)
    feature2 = st.number_input("Average RTT",value=0.00)
    feature3 = st.number_input("Average Throughput",value=0.00)

    input_data = np.array([[feature1,feature2,feature3]])

    if st.button("predict"):
        prediction = predict(input_data)
        st.success(f"prediction value is : {prediction[0]}")





if __name__ == "__main__":
    main()


