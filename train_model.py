import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset1.csv")

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'name', 'spec_rating'])

# Handle categorical encoding
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols)

# Features and target
X = df.drop(columns=['price'])
y = df['price']

# Save the column order for prediction
columns = X.columns.tolist()
with open("columns.pkl", "wb") as f:
    pickle.dump(columns, f)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model training completed and saved as rf_model.pkl")
