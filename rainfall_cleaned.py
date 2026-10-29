


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("Current working directory:", os.getcwd())

file_path = "rainfall_cleaned.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "rainfall_cleaned.csv was not found in the current folder. "
        "Set the working directory to C:\\Users\\kpran\\Rainfall_CA_2."
    )

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================

df.columns = df.columns.str.strip()
df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.strip()

df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
df["ANNUAL"] = pd.to_numeric(df["ANNUAL"], errors="coerce")

# Remove rows with missing year or target
df = df.dropna(subset=["YEAR", "ANNUAL"])
df["YEAR"] = df["YEAR"].astype(int)

# Remove negative annual rainfall
df = df[df["ANNUAL"] >= 0].copy()

# Keep only one record for each subdivision and year
df = df.drop_duplicates(subset=["SUBDIVISION", "YEAR"])

# Sort data chronologically
df = df.sort_values(["SUBDIVISION", "YEAR"])

print("\nDataset shape after cleaning:", df.shape)

# ============================================================
# STEP 3: FEATURE ENGINEERING
# Predict rainfall using previous-year rainfall
# ============================================================

df["PREVIOUS_YEAR_RAINFALL"] = (
    df.groupby("SUBDIVISION")["ANNUAL"].shift(1)
)

# Remove rows without a previous-year observation
model_data = df.dropna(
    subset=["PREVIOUS_YEAR_RAINFALL"]
).copy()

features = ["PREVIOUS_YEAR_RAINFALL", "SUBDIVISION"]
target = "ANNUAL"

print("Records available for modelling:", len(model_data))

# ============================================================
# STEP 4: TRAIN-TEST SPLIT BY YEAR
# ============================================================

split_year = 2007

train = model_data[model_data["YEAR"] <= split_year].copy()
test = model_data[model_data["YEAR"] > split_year].copy()

if train.empty or test.empty:
    raise ValueError(
        "Training or testing data is empty. Check the YEAR values."
    )

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\nTraining records:", len(train))
print("Testing records:", len(test))
print("Training period:",
      train["YEAR"].min(), "-", train["YEAR"].max())
print("Testing period:",
      test["YEAR"].min(), "-", test["YEAR"].max())

# ============================================================
# STEP 5: PREPROCESSING
# ============================================================

numeric_features = ["PREVIOUS_YEAR_RAINFALL"]
categorical_features = ["SUBDIVISION"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# ============================================================
# STEP 6: DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),

    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# ============================================================
# STEP 7: TRAIN AND EVALUATE MODELS
# ============================================================

results = []
predictions = {}

for name, model in models.items():

    print("\nTraining:", name)

    # Train model using earlier years
    model.fit(X_train, y_train)

    # Predict rainfall for later years
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

# Display results
results_df = pd.DataFrame(results).sort_values("MAE")

print("\n========== MODEL PERFORMANCE ==========")
print(results_df.to_string(index=False))

# ============================================================
# STEP 8: COMPARE MODEL PERFORMANCE
# ============================================================

plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["MAE"])

plt.title("Comparison of Rainfall Prediction Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Mean Absolute Error (mm)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ============================================================
# STEP 9: ACTUAL VS PREDICTED RAINFALL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]
best_predictions = predictions[best_model_name]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_predictions, alpha=0.5)

lower = min(y_test.min(), best_predictions.min())
upper = max(y_test.max(), best_predictions.max())

plt.plot([lower, upper], [lower, upper], "r--")

plt.title("Actual vs Predicted Rainfall: " + best_model_name)
plt.xlabel("Actual Annual Rainfall (mm)")
plt.ylabel("Predicted Annual Rainfall (mm)")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 10: SAVE RESULTS
# ============================================================

results_df.to_csv("rainfall_model_results.csv", index=False)

prediction_results = test[
    ["SUBDIVISION", "YEAR", "ANNUAL"]
].copy()

prediction_results["PREDICTED_ANNUAL"] = best_predictions

prediction_results.to_csv("rainfall_predictions.csv", index=False)

print("\nBest model by MAE:", best_model_name)
print("Saved: rainfall_model_results.csv")
print("Saved: rainfall_predictions.csv")
print("Output folder:", os.getcwd())

print("\nMachine Learning implementation completed successfully!")
