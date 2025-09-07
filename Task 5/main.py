import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from packaging import version
import sklearn

# -----------------------------
# 1. Load CSV
# -----------------------------
df = pd.read_csv("housing_data.csv")

# Target
y = df["price"].values

# Features
num_features = ["sqft", "bedrooms", "bathrooms", "year_built"]
cat_features = ["neighborhood"]

# One-hot encode categorical (fix for sklearn>=1.2)
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    encoder = OneHotEncoder(sparse_output=False)
else:
    encoder = OneHotEncoder(sparse=False)

X_cat = encoder.fit_transform(df[cat_features])

# Scale numeric
scaler = StandardScaler()
X_num = scaler.fit_transform(df[num_features])

# Final tabular features
X_tab = np.hstack([X_num, X_cat])

# -----------------------------
# 2. Load Images
# -----------------------------
img_size = (128, 128)
images = []

for path in df["image_path"]:
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        images.append(img)
    else:
        # blank image if missing
        images.append(np.zeros((*img_size, 3)))

X_img = np.array(images)

# -----------------------------
# 3. Train/Test Split
# -----------------------------
X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
    X_tab, X_img, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. CNN for Images
# -----------------------------
img_input = Input(shape=(128, 128, 3))
x = layers.Conv2D(32, (3, 3), activation="relu")(img_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
img_model = models.Model(inputs=img_input, outputs=x)

# -----------------------------
# 5. MLP for Tabular
# -----------------------------
tab_input = Input(shape=(X_tab.shape[1],))
y1 = layers.Dense(32, activation="relu")(tab_input)
y1 = layers.Dense(16, activation="relu")(y1)
tab_model = models.Model(inputs=tab_input, outputs=y1)

# -----------------------------
# 6. Combine Models
# -----------------------------
combined = layers.concatenate([img_model.output, tab_model.output])
z = layers.Dense(64, activation="relu")(combined)
z = layers.Dense(1)(z)

model = models.Model(inputs=[img_model.input, tab_model.input], outputs=z)

# -----------------------------
# 7. Compile & Train
# -----------------------------
model.compile(optimizer="adam", loss="mse")

history = model.fit(
    [X_img_train, X_tab_train], y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=4,
    verbose=1
)

# -----------------------------
# 8. Evaluate
# -----------------------------
y_pred = model.predict([X_img_test, X_tab_test])

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("✅ MAE:", mae)
print("✅ RMSE:", rmse)
