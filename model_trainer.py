import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
df = pd.read_csv('irrigation_prediction.csv')

# 2. Preprocessing
# We drop the target and convert all text columns to numbers automatically
X = df.drop('Irrigation_Need', axis=1)
X = pd.get_dummies(X) # One-Hot Encoding
y = df['Irrigation_Need']

# CRITICAL: Save the column names so the App knows the exact order
model_columns = list(X.columns)
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)

# 3. Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save Model
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained successfully! Accuracy: {model.score(X_test, y_test)*100:.2f}%")
