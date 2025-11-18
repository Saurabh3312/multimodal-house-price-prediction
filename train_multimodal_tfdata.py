# train_multimodal_tfdata.py
# Multimodal model: Image (MobileNetV2) + Tabular -> Fusion -> Regression (price)
# Uses TF.DATA pipeline → VERY FAST, NO RAM ISSUES

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

CSV_PATH = "socal2.csv"
IMAGES_DIR = "images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# -------------------------------------
# 1) Load CSV
# -------------------------------------
df = pd.read_csv(CSV_PATH)

# Preprocess tabular data
tabular_cols = ["n_citi", "bed", "bath", "sqft"]
if "citi" in df.columns:
    le_city = LabelEncoder()
    df["citi_encoded"] = le_city.fit_transform(df["citi"])
    tabular_cols.append("citi_encoded")

X_tab = df[tabular_cols].values
y = df["price"].values
image_ids = df["image_id"].values

# Train-test split
(X_tab_train, X_tab_test,
 y_train, y_test,
 image_ids_train, image_ids_test) = train_test_split(
    X_tab, y, image_ids, test_size=0.2, random_state=42
)

# Scale tabular
scaler = StandardScaler()
X_tab_train = scaler.fit_transform(X_tab_train)
X_tab_test = scaler.transform(X_tab_test)

# Convert tabular data to tf tensors
X_tab_train_tf = tf.constant(X_tab_train, dtype=tf.float32)
X_tab_test_tf = tf.constant(X_tab_test, dtype=tf.float32)

# -------------------------------------
# 2) TF.DATA IMAGE PIPELINE
# -------------------------------------
def load_image_tf(image_id, tab_data, label):
    img_path = tf.strings.join([IMAGES_DIR, "/", tf.strings.as_string(image_id), ".jpg"])
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3, try_recover_truncated=True)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return (tab_data, img), label

# Create tf.data dataset
train_ds = tf.data.Dataset.from_tensor_slices((image_ids_train, X_tab_train_tf, y_train))
train_ds = train_ds.map(load_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((image_ids_test, X_tab_test_tf, y_test))
test_ds = test_ds.map(load_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------------------
# 3) Build the multimodal model
# -------------------------------------
# Tabular branch
tab_input = Input(shape=(X_tab_train.shape[1],), name="tab_input")
x = Dense(128, activation="relu")(tab_input)
x = Dense(64, activation="relu")(x)
tab_out = Dense(32, activation="relu")(x)

# Image branch
img_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="img_input")
base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
base.trainable = False  # freeze
img_feat = base(img_input)
img_x = Dense(128, activation="relu")(img_feat)
img_out = Dense(32, activation="relu")(img_x)

# Fusion
combined = Concatenate()([tab_out, img_out])
z = Dense(128, activation="relu")(combined)
z = Dense(64, activation="relu")(z)
output = Dense(1)(z)

model = Model(inputs=[tab_input, img_input], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.summary()

# -------------------------------------
# 4) Train the model
# -------------------------------------
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# -------------------------------------
# 5) Evaluate
# -------------------------------------
preds = model.predict(test_ds).flatten()
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)