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