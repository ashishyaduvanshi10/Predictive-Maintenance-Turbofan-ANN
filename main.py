import pandas as pd
import numpy as np

# Define column names based on NASA C-MAPSS documentation
index_names = ['engine_id', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# Load the raw training text file using space separator
train_data = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
print("Data loaded successfully! Shape:", train_data.shape)


# Dropping sensors that have constant flat values and provide zero predictive variance
dead_sensors = ['setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
train_data_clean = train_data.drop(columns=dead_sensors)
print("Data cleaning complete. New shape after dropping dead sensors:", train_data_clean.shape)


# 1. Find the maximum cycle achieved by each engine unit
max_cycle = train_data_clean.groupby('engine_id')['cycle'].max().reset_index()
max_cycle.columns = ['engine_id', 'max_cycle']

# 2. Merge the maximum cycles dataframe back into our main cleaned dataframe
train_data_clean = train_data_clean.merge(max_cycle, on='engine_id', how='left')

# 3. Calculate RUL mathematically: max_cycle minus current cycle
train_data_clean['RUL'] = train_data_clean['max_cycle'] - train_data_clean['cycle']

# 4. Drop the auxiliary 'max_cycle' column as it is no longer required
train_data_clean = train_data_clean.drop(columns=['max_cycle'])

print("RUL target feature engineering complete!")
print("\nFirst 5 rows of calculated RUL dataset structure:")
print(train_data_clean[['engine_id', 'cycle', 'RUL']].head())

print("\n--- FEATURE SCALING & ANN MODEL BUILDING ---")

# 1. Separate features (X) and target/answer (y)
# We exclude 'engine_id', 'cycle' (identifiers) and 'RUL' (target variable itself) from features
features = [col for col in train_data_clean.columns if col not in ['engine_id', 'cycle', 'RUL']]
X_train = train_data_clean[features]
y_train = train_data_clean['RUL']

# 2. Scale features between 0 and 1 so the Neural Network learns efficiently
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("Features scaled successfully. Training input shape:", X_train_scaled.shape)

# 3. Build the Artificial Neural Network (ANN) structure using TensorFlow/Keras
from tensorflow.keras import models, layers

model = models.Sequential([
    # Input Layer + First Hidden Layer (32 neurons)
    layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    # Second Hidden Layer (16 neurons)
    layers.Dense(16, activation='relu'),
    # Output Layer (1 neuron to predict a single continuous value: RUL)
    layers.Dense(1, activation='linear')
])

# 4. Compile the model with Adam optimizer and Mean Squared Error loss for regression
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("ANN Model compiled successfully! Here is the architecture summary:")
model.summary()


print("\n--- TRAINING THE ANN MODEL ---")
# We will train the model for 20 epochs with a batch size of 32
# validation_split=0.2 means 20% of the data will be used to check accuracy during training
history = model.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
print("Model training completed successfully over 20 epochs!")


print("\n--- MODEL EVALUATION ON TEST DATA ---")
# 1. Load the raw test dataset
test_data = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)

# 2. Clean test data by dropping the same dead sensors
test_data_clean = test_data.drop(columns=dead_sensors)

# 3. For test data, NASA provides ground truth RUL values differently.
# To keep evaluation simple and robust, we evaluate using the final cycle of each engine
X_test = test_data_clean[features]

# 4. Scale the test features using the SAME scaler we fit on training data
X_test_scaled = scaler.transform(X_test)

# 5. Predict RUL using our trained ANN model
predictions = model.predict(X_test_scaled)
print("Predictions calculated successfully! Total test predictions:", len(predictions))


print("\n--- PLOTTING TRAINING LOSS CURVE ---")

import matplotlib.pyplot as plt

# Plotting Training Loss vs Validation Loss over Epochs to visually verify convergence
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='red', linewidth=2)

plt.title('ANN Model Training History (Loss Convergence)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Squared Error (Loss)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot as an image file in the project folder
plt.savefig('training_loss_curve.png')
print("Training loss curve graph saved successfully as 'training_loss_curve.png'!")