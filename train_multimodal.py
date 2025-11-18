# train_multimodal.py
# Multimodal model: Image (MobileNetV2) + Tabular -> Fusion -> Regression (price)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---------- USER SETTINGS ----------
CSV_PATH = "socal2.csv"
IMAGES_DIR = "images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
RANDOM_STATE = 42
MODEL_SAVE_PATH = "multimodal_house_price_model.h5"
# -----------------------------------

# ---------- 1) Load CSV ----------
df = pd.read_csv(CSV_PATH)
print("Rows:", len(df))
print(df.head())

# ---------- 2) Basic Preprocessing for tabular data ----------
# Keep columns: image_id, bed, bath, sqft, n_citi (adjust names if different)
# Drop columns you don't want as direct numeric features (like street).
tabular_cols = ["image_id", "n_citi", "bed", "bath", "sqft"]
for c in tabular_cols:
    if c not in df.columns:
        raise ValueError(f"Expected column '{c}' in CSV but not found.")

# Optionally encode 'citi' or 'street' as features if you want (example below)
if "citi" in df.columns:
    le_city = LabelEncoder()
    df["citi_encoded"] = le_city.fit_transform(df["citi"])
    tabular_cols.append("citi_encoded")

# Fill missing numeric values (simple strategy)
df[["n_citi", "bed", "bath", "sqft"]] = df[["n_citi", "bed", "bath", "sqft"]].fillna(df[["n_citi", "bed", "bath", "sqft"]].median())

# Target
y = df["price"].values

# Tabular features (remove image_id from features for scaling separately)
X_tab = df[tabular_cols].copy()
image_ids = X_tab["image_id"].astype(int).values
X_tab = X_tab.drop(columns=["image_id"]).values  # shape: (n, num_tab_features)

# Train-test split (keep images & tabular aligned)
(X_tab_train, X_tab_test,
 image_ids_train, image_ids_test,
 y_train, y_test) = train_test_split(
    X_tab, image_ids, y, test_size=0.2, random_state=RANDOM_STATE
)

# Scale tabular features
scaler = StandardScaler()
X_tab_train = scaler.fit_transform(X_tab_train)
X_tab_test = scaler.transform(X_tab_test)

# ---------- 3) Image loading helper ----------
def load_and_preprocess_image(img_id):
    path = os.path.join(IMAGES_DIR, f"{int(img_id)}.jpg")
    if not os.path.exists(path):
        # If image missing, return a zero image (or you can return a random / placeholder)
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    img = load_img(path, target_size=IMG_SIZE)  # PIL image
    arr = img_to_array(img) / 255.0
    return arr.astype(np.float32)

# Prepare image numpy arrays (you could also build tf.data for streaming large datasets)
X_img_train = np.array([load_and_preprocess_image(i) for i in image_ids_train])
X_img_test = np.array([load_and_preprocess_image(i) for i in image_ids_test])

print("Tabular train shape:", X_tab_train.shape)
print("Image train shape:", X_img_train.shape)
print("y train shape:", y_train.shape)

# ---------- 4) Build the multimodal model ----------
# Tabular branch
tab_input = Input(shape=(X_tab_train.shape[1],), name="tab_input")
x = Dense(128, activation="relu")(tab_input)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)
tab_output = Dense(32, activation="relu")(x)

# Image branch - pretrained MobileNetV2
img_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="img_input")
base_cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_cnn.trainable = False  # freeze base initially
img_features = base_cnn(img_input)
img_x = Dense(256, activation="relu")(img_features)
img_x = Dropout(0.3)(img_x)
img_output = Dense(64, activation="relu")(img_x)

# Fusion
combined = Concatenate()([tab_output, img_output])
z = Dense(128, activation="relu")(combined)
z = Dropout(0.25)(z)
z = Dense(64, activation="relu")(z)
z = Dense(32, activation="relu")(z)
output = Dense(1, activation="linear", name="price_output")(z)

model = Model(inputs=[tab_input, img_input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="mse",
              metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

model.summary()

# ---------- 5) Callbacks ----------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1)
]

# ---------- 6) Train ----------
history = model.fit(
    x=[X_tab_train, X_img_train],
    y=y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save final model (in addition to checkpoint)
model.save(MODEL_SAVE_PATH)
print("Model saved to", MODEL_SAVE_PATH)

# ---------- 7) Evaluation ----------
preds = model.predict([X_tab_test, X_img_test]).flatten()
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print("TEST MAE:", mae)
print("TEST RMSE:", rmse)
print("TEST R2:", r2)

# Plot training curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history['mean_absolute_error'], label='train_mae')
plt.plot(history.history['val_mean_absolute_error'], label='val_mae')
plt.legend(); plt.title("MAE")
plt.tight_layout()
plt.show()

# Visualize some predictions
n_display = min(10, len(preds))
plt.figure(figsize=(6,6))
plt.scatter(y_test[:n_display], preds[:n_display])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (first rows)")
plt.show()
