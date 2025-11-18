# train_multimodal_final.py
# Final stable multimodal training script
# Requirements: tensorflow, pandas, scikit-learn, matplotlib, numpy
# Run with Python 3.11 (tensorflow compatible interpreter)

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import datetime

# ---------------- CONFIG ----------------
CSV_PATH = "houses_dataset_final_fixed.csv"   # final CSV (fixed ids)
IMAGES_ROOT = "images_final"
HOUSE_ID_COL = "house_id"
TARGET_COL = "price"
TABULAR_COLS = ["bed", "bath", "sqft"]       # use columns present in CSV
IMG_SIZE = (224, 224)
K_INDOOR = 3
BATCH_SIZE = 8
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 12
MODEL_PATH = "multimodal_final_best.keras"
LOG_TARGET = True    # train on log(price)
SEED = 42
TF_AUTOTUNE = tf.data.AUTOTUNE
RANDOM_AUG = True    # additional lightweight augmentation
UNFREEZE_TOP_N = 60  # top layers of MobileNet to unfreeze during fine-tune
# ----------------------------------------

# reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# helpful prints
print("Running training with config:")
print(f" CSV: {CSV_PATH}")
print(f" Images root: {IMAGES_ROOT}")
print(f" K_INDOOR: {K_INDOOR}, IMG_SIZE: {IMG_SIZE}, BATCH: {BATCH_SIZE}")
print()

# ---------------- 1) Load CSV ----------------
if not os.path.exists(CSV_PATH):
    raise SystemExit(f"CSV not found: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print("CSV loaded. Rows:", len(df))
print(df.head())

# ---------------- 2) Build dataset from folders (robust) ----------------
folders = sorted([d for d in os.listdir(IMAGES_ROOT) if os.path.isdir(os.path.join(IMAGES_ROOT, d))])
print("Found folders:", len(folders), "sample:", folders[:10])

# index csv rows by string house_id
df_index = df.set_index(df[HOUSE_ID_COL].astype(str)).to_dict(orient="index")

rows = []
for folder in folders:
    folder_path = os.path.join(IMAGES_ROOT, folder)

    # find outdoor (common names) else pick first image
    common_out = ["outdoor.jpg", "outdoor.jpeg", "outdoor.png", "front.jpg", "exterior.jpg"]
    outdoor = None
    for name in common_out:
        p = os.path.join(folder_path, name)
        if os.path.exists(p):
            outdoor = p
            break
    if outdoor is None:
        imgs = sorted([p for p in glob.glob(os.path.join(folder_path, "*.*")) if p.lower().endswith((".jpg",".jpeg",".png"))])
        if not imgs:
            print("No images in folder (skipping):", folder_path)
            continue
        outdoor = imgs[0]

    # indoor images = other images in folder
    all_imgs = sorted([p for p in glob.glob(os.path.join(folder_path, "*.*")) if p.lower().endswith((".jpg",".jpeg",".png"))])
    indoor_imgs = [p for p in all_imgs if p != outdoor]
    if not indoor_imgs:
        indoor_imgs = [outdoor]
    # pad or trim
    if len(indoor_imgs) < K_INDOOR:
        while len(indoor_imgs) < K_INDOOR:
            indoor_imgs.append(indoor_imgs[-1])
    else:
        indoor_imgs = indoor_imgs[:K_INDOOR]

    # match CSV metadata by folder id string
    if folder not in df_index:
        # warn and skip if not present in csv
        print("Folder not found in CSV (skipping):", folder)
        continue
    meta = df_index[folder]
    if TARGET_COL not in meta or pd.isna(meta[TARGET_COL]):
        print("Missing target price for house (skipping):", folder)
        continue

    row = {"house_id": folder, "out_path": outdoor}
    for i in range(K_INDOOR):
        row[f"indoor_{i}"] = indoor_imgs[i]
    # add tabular columns, fill missing with 0
    for c in TABULAR_COLS:
        row[c] = meta.get(c, 0)
    row[TARGET_COL] = meta[TARGET_COL]
    rows.append(row)

proc = pd.DataFrame(rows).reset_index(drop=True)
print("Prepared dataset rows:", len(proc))
if len(proc) == 0:
    raise SystemExit("No prepared rows. Check CSV and images.")

# ---------------- 3) Prepare arrays and split ----------------
X_tab = proc[TABULAR_COLS].astype(np.float32).values
y = proc[TARGET_COL].astype(np.float32).values
out_paths = proc["out_path"].values
indoor_cols = [f"indoor_{i}" for i in range(K_INDOOR)]
indoor_paths = proc[indoor_cols].values.astype(str)

X_tab_train, X_tab_test, out_train, out_test, ind_train, ind_test, y_train, y_test = train_test_split(
    X_tab, out_paths, indoor_paths, y, test_size=0.20, random_state=SEED
)

# scale tabular
scaler = StandardScaler()
X_tab_train = scaler.fit_transform(X_tab_train)
X_tab_test = scaler.transform(X_tab_test)

# log-transform target if requested
if LOG_TARGET:
    y_train_t = np.log1p(y_train)
    y_test_t  = np.log1p(y_test)
else:
    y_train_t, y_test_t = y_train, y_test

print("Train samples:", len(y_train), "Val samples:", len(y_test))

# ---------------- 4) tf.data pipelines ----------------
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, try_recover_truncated=True)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.06)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    # slight rotation via tf.image.rot90 with random times (small)
    if RANDOM_AUG:
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
    return img

def map_fn(out_p, ind_p, tab, label, training):
    out_img = load_and_preprocess_image(out_p)
    if training:
        out_img = augment_image(out_img)
    # indoor stack
    ind_imgs = tf.stack([load_and_preprocess_image(p) for p in tf.unstack(ind_p)])
    if training:
        ind_imgs = tf.map_fn(lambda x: augment_image(x), ind_imgs, fn_output_signature=tf.float32)
    return (tab, out_img, ind_imgs), label

def make_dataset(out_paths_arr, ind_arr, tab_arr, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((out_paths_arr, ind_arr, tab_arr, labels))
    ds = ds.map(lambda o,i,t,y: map_fn(o,i,t,y, training=training), num_parallel_calls=TF_AUTOTUNE)
    if training:
        ds = ds.shuffle(512)
    ds = ds.batch(BATCH_SIZE).prefetch(TF_AUTOTUNE)
    return ds

train_ds = make_dataset(out_train, ind_train, X_tab_train, y_train_t, training=True)
val_ds = make_dataset(out_test, ind_test, X_tab_test, y_test_t, training=False)

# ---------------- 5) Model architecture ----------------
# Tabular branch
tab_in = tf.keras.Input(shape=(X_tab_train.shape[1],), name="tab_input")
x = tf.keras.layers.Dense(128, activation="relu")(tab_in)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
tab_out = tf.keras.layers.Dense(32, activation="relu")(x)

# Outdoor branch
out_in = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="out_input")
base_out = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg")
base_out.trainable = False
o = base_out(out_in)
o = tf.keras.layers.Dense(128, activation="relu")(o)
out_feat = tf.keras.layers.Dense(32, activation="relu")(o)

# Indoor branch (K images)
ind_in = tf.keras.Input(shape=(K_INDOOR, IMG_SIZE[0], IMG_SIZE[1], 3), name="indoor_input")
base_ind = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg")
base_ind.trainable = False
td = tf.keras.layers.TimeDistributed(base_ind)(ind_in)
td = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu"))(td)
ind_agg = tf.keras.layers.GlobalAveragePooling1D()(td)
ind_feat = tf.keras.layers.Dense(32, activation="relu")(ind_agg)

# Fusion and head
merged = tf.keras.layers.Concatenate()([tab_out, out_feat, ind_feat])
z = tf.keras.layers.Dense(256, activation="relu")(merged)
z = tf.keras.layers.Dropout(0.3)(z)
z = tf.keras.layers.Dense(128, activation="relu")(z)
z = tf.keras.layers.Dense(64, activation="relu")(z)
pred = tf.keras.layers.Dense(1, activation="linear")(z)

model = tf.keras.Model(inputs=[tab_in, out_in, ind_in], outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
model.summary()

# ---------------- 6) Train initial head ----------------
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=logdir)
]

print("\n--- Training head (base CNN frozen) ---")
history = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, callbacks=callbacks)

# Evaluate before fine-tuning
preds_log = model.predict(val_ds).flatten()
if LOG_TARGET:
    preds_before = np.expm1(preds_log)
else:
    preds_before = preds_log

mae_before = mean_absolute_error(y_test, preds_before)
rmse_before = np.sqrt(mean_squared_error(y_test, preds_before))
r2_before = r2_score(y_test, preds_before)

print("\n--- RESULTS BEFORE FINE-TUNING ---")
print("MAE :", mae_before)
print("RMSE:", rmse_before)
print("R2  :", r2_before)

# save plots of history
plt.figure(); plt.plot(history.history['loss'], label='train_loss'); plt.plot(history.history['val_loss'], label='val_loss'); plt.legend(); plt.title('Loss'); plt.savefig('loss_head.png')
plt.figure(); plt.plot(history.history.get('mae', []), label='train_mae'); plt.plot(history.history.get('val_mae', []), label='val_mae'); plt.legend(); plt.title('MAE'); plt.savefig('mae_head.png')

# ---------------- 7) Fine-tuning: unfreeze top layers ----------------
print("\n--- Fine-tuning: unfreezing top layers ---")
# Unfreeze last UNFREEZE_TOP_N layers of base_out and base_ind
def unfreeze_top(model_base, n_top):
    if n_top <= 0:
        return
    total = len(model_base.layers)
    start = max(0, total - n_top)
    for layer in model_base.layers[start:]:
        layer.trainable = True
    print(f"Unfroze {total - start} layers out of {total} for base: {model_base.name}")

unfreeze_top(base_out, UNFREEZE_TOP_N)
unfreeze_top(base_ind, UNFREEZE_TOP_N)

# recompile with lower LR
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="mse", metrics=["mae"])

print("\n--- Training (fine-tune) ---")
history_ft = model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                                  tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)])

# Evaluate after fine-tuning
preds_log_ft = model.predict(val_ds).flatten()
if LOG_TARGET:
    preds_after = np.expm1(preds_log_ft)
else:
    preds_after = preds_log_ft

mae_after = mean_absolute_error(y_test, preds_after)
rmse_after = np.sqrt(mean_squared_error(y_test, preds_after))
r2_after = r2_score(y_test, preds_after)

print("\n--- RESULTS AFTER FINE-TUNING ---")
print("MAE :", mae_after)
print("RMSE:", rmse_after)
print("R2  :", r2_after)

# save final plot
plt.figure(); plt.plot(history.history['loss'], label='head_loss'); plt.plot(history.history['val_loss'], label='head_val_loss')
plt.plot(np.arange(len(history.history['loss']), len(history.history['loss'])+len(history_ft.history['loss'])),
         history_ft.history['loss'], label='ft_loss')
plt.plot(np.arange(len(history.history['val_loss']), len(history.history['val_loss'])+len(history_ft.history['val_loss'])),
         history_ft.history['val_loss'], label='ft_val_loss')
plt.legend(); plt.title('Loss combined'); plt.savefig('loss_combined.png')

plt.figure(); 
plt.plot(history.history.get('mae', []), label='head_mae')
plt.plot(history.history.get('val_mae', []), label='head_val_mae')
if history_ft.history.get('mae'): 
    plt.plot(np.arange(len(history.history.get('mae', [])), len(history.history.get('mae', []))+len(history_ft.history.get('mae', []))),
             history_ft.history.get('mae', []), label='ft_mae')
plt.legend(); plt.title('MAE combined'); plt.savefig('mae_combined.png')

# Save model (best was saved by checkpoint, but save final weights)
model.save("multimodal_final_last.keras")
print("Saved final model to multimodal_final_last.h5 and best checkpoint to", MODEL_PATH)

# ---------------- 8) Grad-CAM utility (for outdoor branch) ----------------
def make_gradcam_on_image(model, image_path, class_idx=None, layer_name=None, save_to="gradcam_outdoor.png"):
    """
    Simple Grad-CAM for the outdoor branch (visualize which parts contributed).
    Default layer_name will be the last conv layer in base_out if not provided.
    """
    img = load_and_preprocess_image(image_path)
    img_batch = tf.expand_dims(img, axis=0)

    # find a convolutional layer in base_out to target
    if layer_name is None:
        # heuristic: last layer which is Conv2D in the base_out model
        for l in reversed(base_out.layers):
            if isinstance(l, tf.keras.layers.Conv2D) or 'conv' in l.name:
                layer_name = l.name
                break
    if layer_name is None:
        print("No conv layer found for grad-cam")
        return

    grad_model = tf.keras.models.Model([model.inputs], [base_out.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([np.zeros((1,)+IMG_SIZE+(3,)),])  # dummy for shape inference
    # Instead we will build a small custom gradcam for single input:
    grad_model2 = tf.keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])

    # run on real image
    conv_outputs, preds = grad_model2([np.zeros((1,)+IMG_SIZE+(3,)), np.zeros((1,)+IMG_SIZE+(3,)), np.zeros((1, K_INDOOR, IMG_SIZE[0], IMG_SIZE[1], 3))])
    # NOTE: this helper is intentionally simple; for rigorous Grad-CAM we would re-build a model that exposes the conv layer.
    print("Grad-CAM helper available — use a dedicated snippet for target layer extraction if needed.")

# Quick sample grad-cam save (optional)
try:
    sample_folder = proc['house_id'].iloc[0]
    sample_out = proc['out_path'].iloc[0]
    print("Sample house for Grad-CAM:", sample_folder, sample_out)
    # You can call make_gradcam_on_image(sample_out) manually if you want; function is a placeholder helper.
except Exception:
    pass

# ---------------- 9) Final printed summary ----------------
print("\n=== FINAL SUMMARY ===")
print("Rows prepared:", len(proc))
print("Before FT -> MAE:", mae_before, "RMSE:", rmse_before, "R2:", r2_before)
print("After FT  -> MAE:", mae_after, "RMSE:", rmse_after, "R2:", r2_after)
print("Model saved as:", MODEL_PATH, "and multimodal_final_last.keras")
print("Loss/MAE plots: loss_head.png, mae_head.png, loss_combined.png, mae_combined.png")
print("====================================\n")

# End of script
