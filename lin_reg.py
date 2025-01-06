import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('StudentsPerformance.csv')

# Select feature and target
X = df[['reading score']]
y = df['math score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit app
st.title("Linear Regression on Student Performance")
st.write("This app predicts math scores based on reading scores using a linear regression model.")

# Display dataset
st.subheader("Dataset")
st.write(df.head())

# Display metrics
st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R^2 Score: {r2}")

# Plot regression line
st.subheader("Regression Line")
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='blue', label='Data points')
ax.plot(X_test, y_pred, color='red', label='Regression line')
ax.set_xlabel('Reading Score')
ax.set_ylabel('Math Score')
ax.set_title('Linear Regression')
ax.legend()
st.pyplot(fig)
