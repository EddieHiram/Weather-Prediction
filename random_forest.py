
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('dataset.csv')

# Data Preprocessing

# 1. Handle missing values
data.fillna(data.mean(), inplace=True)  # Fills missing numeric values with the mean

# 2. Select Features and Target
# We're assuming TAVG is the target variable for prediction
X = data.drop(['TAVG', 'DATE', 'STATION', 'NAME'], axis=1, errors='ignore')
y = data['TAVG']

# 3. Encode categorical features and scale numerical features
# Assume categorical features are 'WT01' and 'WT08' (weather types)
categorical_features = ['WT01', 'WT08']
numerical_features = X.columns.difference(categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline with preprocessing and RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Feature Importance (using the model in the pipeline)
import matplotlib.pyplot as plt

# Extract feature importances
feature_importances = pipeline.named_steps['regressor'].feature_importances_

# Get feature names after preprocessing
encoded_features = list(numerical_features) + list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
feature_importances_series = pd.Series(feature_importances, index=encoded_features)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances_series.nlargest(10).plot(kind='barh')
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features")
plt.show()
