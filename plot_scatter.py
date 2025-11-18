import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import exp
from train_multimodal_final import create_dataset, build_model  # if needed adjust imports

CSV_PATH = "houses_dataset_final_fixed.csv"
MODEL_PATH = "multimodal_final_best.keras"
IMAGES_ROOT = "images_final"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Reload dataset using the same preprocessing function
from train_multimodal_final import prepare_dataset
proc = prepare_dataset(df)

# Select features & labels
X_tab = np.stack(proc["tab"], axis=0)
X_out = np.stack(proc["outdoor"], axis=0)
X_in = np.stack(proc["indoor"], axis=0)
y = np.array(proc["price"])

# Train-test split
from sklearn.model_selection import train_test_split
X_tab_train, X_tab_test, X_out_train, X_out_test, X_in_train, X_in_test, y_train, y_test = train_test_split(
    X_tab, X_out, X_in, y, test_size=0.2, random_state=42
)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Predictions
y_pred = model.predict([X_out_test, X_in_test, X_tab_test])
y_pred = np.expm1(y_pred).flatten()
y_real = np.expm1(y_test)

# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(y_real, y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Figure 4.5: Actual vs Predicted House Prices")
plt.grid(True)
plt.savefig("actual_vs_predicted_scatter.png")
plt.show()
