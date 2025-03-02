import numpy as np # pour les données numériques et tabulaires
import pandas as pd # pour gérer les données sous forme de DataFrame
from sklearn.model_selection import train_test_split # pour diviser les données en ensembles de formation et de test
from sklearn.linear_model import LinearRegression # pour la régression linéaire
from sklearn.metrics import mean_squared_error, r2_score # pour évaluer le modèle
import matplotlib.pyplot as plt # pour visualiser les résultats

data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500, 4000, 4500],
    'Price': [200000, 250000, 300000, 350000, 400000, 500000, 600000]
}
df = pd.DataFrame(data)
print(df.head())
X = df[['SquareFootage']]  # Variables indépendantes (caractéristiques)
y = df['Price']            # Variable dépendante (cible)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Display the learned coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Display the predictions
print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()

# Show the plot
plt.show()


































