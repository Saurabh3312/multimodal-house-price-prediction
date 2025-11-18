# ============================================================
# MULTIMODAL HOUSE PRICE PREDICTION PROJECT (RUN IN VS CODE)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2

# ============================================================
# 1. LOAD DATA
# ===========================================================

df = pd.read_csv("socal2.csv")
print(df.head())
print(df.info())

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Basic statistics
print("\nStatistics:\n", df.describe())

# Correlation heatmap
# Correlation heatmap (Fix for string columns)
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# Price distribution
plt.figure(figsize=(7, 5))
sns.histplot(df['price'], kde=True)
plt.title("Price Distribution")
plt.show()

# Scatter: sqft vs price
plt.figure(figsize=(7, 5))
plt.scatter(df["sqft"], df["price"])
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Sqft vs Price")
plt.show()

# ============================================================
# 3. PREPROCESSING
# ============================================================

# Encode categorical features
label_cols = ["street", "citi"]
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X_tab = df.drop(["price"], axis=1)
y = df["price"]

# Split
X_train_tab, X_test_tab, y_train, y_test = train_test_split(
    X_tab, y, test_size=0.2, random_state=42
)

# Scale numerical data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_tab)
X_test_scaled = scaler.transform(X_test_tab)

# ============================================================
# 4. IMAGE DATA LOADING FUNCTION
# ============================================================

def load_image(img_id):
    path = f"images/{img_id}.jpg"
    if os.path.exists(path):
        img = load_img(path, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        return img
    else:
        print(f"Warning: Missing image {path}")
        return np.zeros((224, 224, 3))

def prepare_image_set(image_ids):
    imgs = np.array([load_image(i) for i in image_ids])
    return imgs

X_train_img = prepare_image_set(X_train_tab["image_id"])
X_test_img = prepare_image_set(X_test_tab["image_id"])

# ============================================================
# 5. BUILD MULTIMODAL MODEL
# ============================================================

# -------- TABULAR BRANCH --------
tab_input = Input(shape=(X_train_scaled.shape[1],))
tab_dense = Dense(128, activation="relu")(tab_input)
tab_dense = Dense(64, activation="relu")(tab_dense)

# -------- IMAGE BRANCH --------
cnn_base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
cnn_base.trainable = False  # Freeze pretrained layers

img_input = Input(shape=(224, 224, 3))
img_features = cnn_base(img_input)
img_dense = Dense(128, activation="relu")(img_features)

# -------- FUSION --------
combined = Concatenate()([tab_dense, img_dense])
combined_dense = Dense(128, activation="relu")(combined)
combined_dense = Dense(64, activation="relu")(combined_dense)
output = Dense(1)(combined_dense)

model = Model(inputs=[tab_input, img_input], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.summary()

# ============================================================
# 6. TRAIN THE MODEL
# ============================================================

history = model.fit(
    [X_train_scaled, X_train_img],
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16
)

# Plot loss
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.title("Training Curve")
plt.show()

# ============================================================
# 7. EVALUATION
# ============================================================

pred = model.predict([X_test_scaled, X_test_img])

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("\n===== Model Evaluation =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# Scatter Plot — Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

# ============================================================
# END
# ============================================================
