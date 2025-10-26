import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    # 1️⃣ Load datasets
    crop_price = pd.read_csv("crop_price.csv")
    crop_suitable = pd.read_csv("crop_suitable.csv")

    # 2️⃣ Fix column names and clean data
    crop_price.rename(columns={
        'State':'state',
        'District':'district',
        'Market':'market',
        'Commodity':'commodity',
        'Modal_x0020_Price':'prices'
    }, inplace=True)

    crop_price = crop_price.dropna(subset=['prices'])
    crop_price['prices'] = pd.to_numeric(crop_price['prices'], errors='coerce')
    crop_price = crop_price.dropna(subset=['prices'])

    # 3️⃣ Average price per crop per market
    avg_price = crop_price.groupby(['state','district','market','commodity'])['prices'].mean().reset_index()

    # 4️⃣ Encode categorical columns
    label_encoders = {}
    for col in ['state','district','market','commodity']:
        label_encoders[col] = LabelEncoder()
        avg_price[col] = label_encoders[col].fit_transform(avg_price[col])

    # 5️⃣ Preprocess crop_suitable
    
    numeric_cols = ['N','P','K','temperature','humidity','ph','rainfall']
    
    scaler = StandardScaler()
    crop_suitable[numeric_cols] = scaler.fit_transform(crop_suitable[numeric_cols])

    # 6️⃣ Merge datasets (cross join)
    merged_data = pd.merge(avg_price, crop_suitable, how="cross")

    # 7️⃣ Define features and target
    feature_cols = numeric_cols + ['state','district','market']
    X = merged_data[feature_cols]
    y = merged_data['prices']

    # 8️⃣ Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 9️⃣ Build TensorFlow model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 10️⃣ Train model
    model.fit(X_train, y_train, epochs=25, batch_size=128, validation_split=0.1)

    # 11️⃣ Save model and preprocessors
    model.save("model.keras",save_format="keras")
    joblib.dump((scaler, label_encoders), "preprocessors.pkl")

    print("✅ Model trained and saved as model.keras")

if __name__ == "__main__":
    train_model()
