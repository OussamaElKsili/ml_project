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
































