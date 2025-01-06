import pandas as pd

# Load the dataset
df = pd.read_csv('StudentsPerformance.csv')

# Display the first few rows of the dataset
print(df.head())

# Select feature and target
X = df[['reading score']]
y = df['math score']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Summary of model performance
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

import matplotlib.pyplot as plt

# Plot the regression line
plt.scatter(X_test, y_test, color='blue', label='Data points')
plt.plot(X_test, y_pred, color='red', label='Regression line')
plt.xlabel('Reading Score')
plt.ylabel('Math Score')
plt.title('Linear Regression')
plt.legend()
plt.show()
