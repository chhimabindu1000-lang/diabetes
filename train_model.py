
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("diabetes.csv")

print("Dataset loaded successfully!")
print(df.head())
print("Dataset shape:", df.shape)

# 2. Separate input and output
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create ML pipeline
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 5. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 6. Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save model
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(list(X.columns), "columns.pkl")

print("\nModel saved successfully!")
