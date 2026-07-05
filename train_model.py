import pandas as pd
import joblib

from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import tqdm

# -------------------------
# Load Dataset
# -------------------------

print("Loading Dataset...")

df = pd.read_csv("dataset/ASAP2_train_sourcetexts.csv")

# -------------------------
# Preprocess Essays
# -------------------------

print("Cleaning Essays...")

tqdm.pandas()

df["clean_text"] = df["full_text"].progress_apply(clean_text)

# -------------------------
# TF-IDF
# -------------------------

print("Creating TF-IDF Features...")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_text"])

y = df["score"]

# -------------------------
# Train Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Random Forest Model
# -------------------------

print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed!")

# -------------------------
# Prediction
# -------------------------

predictions = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.2f}")

print(f"MSE  : {mse:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R² Score : {r2:.2f}")

# -------------------------
# Save Model
# -------------------------

joblib.dump(model, "models/essay_model.pkl")

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")